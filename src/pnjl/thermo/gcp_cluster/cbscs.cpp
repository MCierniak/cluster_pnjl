#include "cbscs.h"

namespace bscs
{

    gsl_function_wrapper cwBDensityBosonSingletIntegrand(cBDensityBosonSingletIntegrand);
    gsl_function_wrapper cwBDensityFermionSingletIntegrand(cBDensityFermionSingletIntegrand);
    gsl_function_wrapper cwBDensityBosonTripletIntegrandReal(cBDensityBosonTripletIntegrandReal);
    gsl_function_wrapper cwBDensityBosonAntitripletIntegrandReal(cBDensityBosonAntitripletIntegrandReal);
    gsl_function_wrapper cwBDensityFermionTripletIntegrandReal(cBDensityFermionTripletIntegrandReal);
    gsl_function_wrapper cwBDensityFermionAntitripletIntegrandReal(cBDensityFermionAntitripletIntegrandReal);
    gsl_function_wrapper cwSDensityBosonSingletIntegrand(cSDensityBosonSingletIntegrand);
    gsl_function_wrapper cwSDensityFermionSingletIntegrand(cSDensityFermionSingletIntegrand);
    gsl_function_wrapper cwSDensityBosonTripletIntegrandReal(cSDensityBosonTripletIntegrandReal);
    gsl_function_wrapper cwSDensityBosonAntitripletIntegrandReal(cSDensityBosonAntitripletIntegrandReal);
    gsl_function_wrapper cwSDensityFermionTripletIntegrandReal(cSDensityFermionTripletIntegrandReal);
    gsl_function_wrapper cwSDensityFermionAntitripletIntegrandReal(cSDensityFermionAntitripletIntegrandReal);

    double cBDensityBosonSingletIntegrand(double p, void *pars)
    {
        cluster_thermo pThArgs = *(cluster_thermo *) pars;
        thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
        double fpi = cFBosonSinglet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs);
        double fmi = cFBosonSinglet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs);
        double fpth = cFBosonSinglet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs);
        double fmth = cFBosonSinglet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs);
        return pow(p, 2)*(fpi - fmi - fpth + fmth);
    }

    double cBDensityFermionSingletIntegrand(double p, void *pars)
    {
        cluster_thermo pThArgs = *(cluster_thermo *) pars;
        thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
        double fpi = cFFermionSinglet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs);
        double fmi = cFFermionSinglet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs);
        double fpth = cFFermionSinglet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs);
        double fmth = cFFermionSinglet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs);
        return pow(p, 2)*(fpi - fmi - fpth + fmth);
    }

    double cBDensityBosonTripletIntegrandReal(double p, void *pars)
    {
        cluster_thermo pThArgs = *(cluster_thermo *) pars;
        thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
        double fpi = cFBosonTriplet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs).real();
        double fmi = cFBosonAntitriplet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs).real();
        double fpth = cFBosonTriplet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs).real();
        double fmth = cFBosonAntitriplet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs).real();
        return pow(p, 2)*(fpi - fmi - fpth + fmth);
    }

    double cBDensityBosonAntitripletIntegrandReal(double p, void *pars)
    {
        cluster_thermo pThArgs = *(cluster_thermo *) pars;
        thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
        double fpi = cFBosonAntitriplet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs).real();
        double fmi = cFBosonTriplet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs).real();
        double fpth = cFBosonAntitriplet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs).real();
        double fmth = cFBosonTriplet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs).real();
        return pow(p, 2)*(fpi - fmi - fpth + fmth);
    }

    double cBDensityFermionTripletIntegrandReal(double p, void *pars)
    {
        cluster_thermo pThArgs = *(cluster_thermo *) pars;
        thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
        double fpi = cFFermionTriplet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs).real();
        double fmi = cFFermionAntitriplet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs).real();
        double fpth = cFFermionTriplet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs).real();
        double fmth = cFFermionAntitriplet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs).real();
        return pow(p, 2)*(fpi - fmi - fpth + fmth);
    }

    double cBDensityFermionAntitripletIntegrandReal(double p, void *pars)
    {
        cluster_thermo pThArgs = *(cluster_thermo *) pars;
        thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
        double fpi = cFFermionAntitriplet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs).real();
        double fmi = cFFermionTriplet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs).real();
        double fpth = cFFermionAntitriplet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs).real();
        double fmth = cFFermionTriplet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs).real();
        return pow(p, 2)*(fpi - fmi - fpth + fmth);
    }

    double cSDensityBosonSingletIntegrand(double p, void *pars)
    {
        cluster_thermo pThArgs = *(cluster_thermo *) pars;
        thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
        double fpi = cFBosonSinglet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs);
        double fmi = cFBosonSinglet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs);
        double fpth = cFBosonSinglet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs);
        double fmth = cFBosonSinglet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs);
        double sigmaPI = (fpi != 0.0) ? fpi*log(fpi) - (1.0 + fpi)*log1p(fpi) : 0.0;
        double sigmaMI = (fmi != 0.0) ? fmi*log(fmi) - (1.0 + fmi)*log1p(fmi) : 0.0;
        double sigmaPTh = (fpth != 0.0) ? fpth*log(fpth) - (1.0 + fpth)*log1p(fpth) : 0.0;
        double sigmaMTh = (fmth != 0.0) ? fmth*log(fmth) - (1.0 + fmth)*log1p(fmth) : 0.0;
        return pow(p, 2)*(sigmaPI + sigmaMI - sigmaPTh - sigmaMTh);
    }

    double cSDensityFermionSingletIntegrand(double p, void *pars)
    {
        cluster_thermo pThArgs = *(cluster_thermo *) pars;
        thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
        double fpi = cFFermionSinglet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs);
        double fmi = cFFermionSinglet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs);
        double fpth = cFFermionSinglet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs);
        double fmth = cFFermionSinglet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs);
        double sigmaPI = (fpi != 0.0) ? fpi*log(fpi) + (1.0 - fpi)*log1p(-fpi) : 0.0;
        double sigmaMI = (fmi != 0.0) ? fmi*log(fmi) + (1.0 - fmi)*log1p(-fmi) : 0.0;
        double sigmaPTh = (fpth != 0.0) ? fpth*log(fpth) + (1.0 - fpth)*log1p(-fpth) : 0.0;
        double sigmaMTh = (fmth != 0.0) ? fmth*log(fmth) + (1.0 - fmth)*log1p(-fmth) : 0.0;
        return pow(p, 2)*(sigmaPI + sigmaMI - sigmaPTh - sigmaMTh);
    }

    double cSDensityBosonTripletIntegrandReal(double p, void *pars)
    {
        cluster_thermo pThArgs = *(cluster_thermo *) pars;
        thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
        particle ppArgsI({pThArgs.mi, pThArgs.chargeB}), paArgsI({pThArgs.mi, -pThArgs.chargeB});
        particle ppArgsTh({pThArgs.mth, pThArgs.chargeB}), paArgsTh({pThArgs.mth, -pThArgs.chargeB});

        double Eni = cEn(p, ppArgsI);
        double fpi = cFBosonTriplet(p, ppArgsI, thArgs).real();
        double fmi = cFBosonAntitriplet(p, paArgsI, thArgs).real();
        double zpi = cZBosonTriplet(p, ppArgsI, thArgs).real();
        double zmi = cZBosonAntitriplet(p, paArgsI, thArgs).real();
        double sigmaPI = ((Eni - ppArgsI.chargeB*thArgs.muB)/thArgs.T)*fpi - zpi/3.0;
        double sigmaMI = ((Eni - paArgsI.chargeB*thArgs.muB)/thArgs.T)*fmi - zmi/3.0;

        double Enth = cEn(p, ppArgsTh);
        double fpth = cFBosonTriplet(p, ppArgsTh, thArgs).real();
        double fmth = cFBosonAntitriplet(p, paArgsTh, thArgs).real();
        double zpth = cZBosonTriplet(p, ppArgsI, thArgs).real();
        double zmth = cZBosonAntitriplet(p, paArgsI, thArgs).real();
        double sigmaPTh = ((Enth - ppArgsTh.chargeB*thArgs.muB)/thArgs.T)*fpth - zpth/3.0;
        double sigmaMTh = ((Enth - paArgsTh.chargeB*thArgs.muB)/thArgs.T)*fmth - zmth/3.0;

        return -pow(p, 2)*(sigmaPI + sigmaMI - sigmaPTh - sigmaMTh);
    }

    double cSDensityBosonAntitripletIntegrandReal(double p, void *pars)
    {
        cluster_thermo pThArgs = *(cluster_thermo *) pars;
        thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
        particle ppArgsI({pThArgs.mi, pThArgs.chargeB}), paArgsI({pThArgs.mi, -pThArgs.chargeB});
        particle ppArgsTh({pThArgs.mth, pThArgs.chargeB}), paArgsTh({pThArgs.mth, -pThArgs.chargeB});

        double Eni = cEn(p, ppArgsI);
        double fpi = cFBosonAntitriplet(p, ppArgsI, thArgs).real();
        double fmi = cFBosonTriplet(p, paArgsI, thArgs).real();
        double zpi = cZBosonAntitriplet(p, ppArgsI, thArgs).real();
        double zmi = cZBosonTriplet(p, paArgsI, thArgs).real();
        double sigmaPI = ((Eni - ppArgsI.chargeB*thArgs.muB)/thArgs.T)*fpi - zpi/3.0;
        double sigmaMI = ((Eni - paArgsI.chargeB*thArgs.muB)/thArgs.T)*fmi - zmi/3.0;

        double Enth = cEn(p, ppArgsTh);
        double fpth = cFBosonAntitriplet(p, ppArgsTh, thArgs).real();
        double fmth = cFBosonTriplet(p, paArgsTh, thArgs).real();
        double zpth = cZBosonAntitriplet(p, ppArgsI, thArgs).real();
        double zmth = cZBosonTriplet(p, paArgsI, thArgs).real();
        double sigmaPTh = ((Enth - ppArgsTh.chargeB*thArgs.muB)/thArgs.T)*fpth - zpth/3.0;
        double sigmaMTh = ((Enth - paArgsTh.chargeB*thArgs.muB)/thArgs.T)*fmth - zmth/3.0;

        return -pow(p, 2)*(sigmaPI + sigmaMI - sigmaPTh - sigmaMTh);
    }

    double cSDensityFermionTripletIntegrandReal(double p, void *pars)
    {
        cluster_thermo pThArgs = *(cluster_thermo *) pars;
        thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
        particle ppArgsI({pThArgs.mi, pThArgs.chargeB}), paArgsI({pThArgs.mi, -pThArgs.chargeB});
        particle ppArgsTh({pThArgs.mth, pThArgs.chargeB}), paArgsTh({pThArgs.mth, -pThArgs.chargeB});

        double Eni = cEn(p, ppArgsI);
        double fpi = cFFermionTriplet(p, ppArgsI, thArgs).real();
        double fmi = cFFermionAntitriplet(p, paArgsI, thArgs).real();
        double zpi = cZFermionTriplet(p, ppArgsI, thArgs).real();
        double zmi = cZFermionAntitriplet(p, paArgsI, thArgs).real();
        double sigmaPI = ((Eni - ppArgsI.chargeB*thArgs.muB)/thArgs.T)*fpi + zpi/3.0;
        double sigmaMI = ((Eni - paArgsI.chargeB*thArgs.muB)/thArgs.T)*fmi + zmi/3.0;

        double Enth = cEn(p, ppArgsTh);
        double fpth = cFFermionTriplet(p, ppArgsTh, thArgs).real();
        double fmth = cFFermionAntitriplet(p, paArgsTh, thArgs).real();
        double zpth = cZFermionTriplet(p, ppArgsI, thArgs).real();
        double zmth = cZFermionAntitriplet(p, paArgsI, thArgs).real();
        double sigmaPTh = ((Enth - ppArgsTh.chargeB*thArgs.muB)/thArgs.T)*fpth + zpth/3.0;
        double sigmaMTh = ((Enth - paArgsTh.chargeB*thArgs.muB)/thArgs.T)*fmth + zmth/3.0;

        return -pow(p, 2)*(sigmaPI + sigmaMI - sigmaPTh - sigmaMTh);
    }

    double cSDensityFermionAntitripletIntegrandReal(double p, void *pars)
    {
        cluster_thermo pThArgs = *(cluster_thermo *) pars;
        thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
        particle ppArgsI({pThArgs.mi, pThArgs.chargeB}), paArgsI({pThArgs.mi, -pThArgs.chargeB});
        particle ppArgsTh({pThArgs.mth, pThArgs.chargeB}), paArgsTh({pThArgs.mth, -pThArgs.chargeB});

        double Eni = cEn(p, ppArgsI);
        double fpi = cFFermionAntitriplet(p, ppArgsI, thArgs).real();
        double fmi = cFFermionTriplet(p, paArgsI, thArgs).real();
        double zpi = cZFermionAntitriplet(p, ppArgsI, thArgs).real();
        double zmi = cZFermionTriplet(p, paArgsI, thArgs).real();
        double sigmaPI = ((Eni - ppArgsI.chargeB*thArgs.muB)/thArgs.T)*fpi + zpi/3.0;
        double sigmaMI = ((Eni - paArgsI.chargeB*thArgs.muB)/thArgs.T)*fmi + zmi/3.0;

        double Enth = cEn(p, ppArgsTh);
        double fpth = cFFermionAntitriplet(p, ppArgsTh, thArgs).real();
        double fmth = cFFermionTriplet(p, paArgsTh, thArgs).real();
        double zpth = cZFermionAntitriplet(p, ppArgsI, thArgs).real();
        double zmth = cZFermionTriplet(p, paArgsI, thArgs).real();
        double sigmaPTh = ((Enth - ppArgsTh.chargeB*thArgs.muB)/thArgs.T)*fpth + zpth/3.0;
        double sigmaMTh = ((Enth - paArgsTh.chargeB*thArgs.muB)/thArgs.T)*fmth + zmth/3.0;

        return -pow(p, 2)*(sigmaPI + sigmaMI - sigmaPTh - sigmaMTh);
    }

    double cMTh(const cluster &hadron, const thermo &thArgs)
    {
        return (hadron.NI - hadron.SI)*cML(thArgs) + hadron.SI*cMS(thArgs);
    }

    double cBDensityCluster(const pi0 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const pi0 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const pi &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const pi &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const K &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const K &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const K0 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const K0 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const eta &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const eta &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const rho &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const rho &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const omega &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const omega &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const Kstar_892 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const Kstar_892 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const Kstar0_892 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const Kstar0_892 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const p &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const p &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const n &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const n &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const etaPrime &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const etaPrime &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const a0 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const a0 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const f0 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const f0 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const phi &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const phi &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const Lambda &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const Lambda &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const h1 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const h1 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const SigmaPlus &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const SigmaPlus &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const Sigma0 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const Sigma0 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const SigmaMinus &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const SigmaMinus &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const b1 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const b1 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const a1 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const a1 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const Delta &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const Delta &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const D1 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonAntitripletIntegrandReal.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const D1 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonAntitripletIntegrandReal.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const D2 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonAntitripletIntegrandReal.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const D2 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonAntitripletIntegrandReal.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const FourQ1 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const FourQ1 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const FourQ2 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const FourQ2 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const FourQ3 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const FourQ3 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const FourQ4 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const FourQ4 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const P1 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const P1 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const P2 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const P2 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const FiveQ1 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityFermionAntitripletIntegrandReal.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const FiveQ1 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityFermionAntitripletIntegrandReal.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const FiveQ2 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityFermionAntitripletIntegrandReal.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const FiveQ2 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityFermionAntitripletIntegrandReal.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const FiveQ3 &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityFermionAntitripletIntegrandReal.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const FiveQ3 &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityFermionAntitripletIntegrandReal.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    double cBDensityCluster(const d &hadron, const thermo &thArgs)
    {
        if (hadron.BI == 0.0)
        {
            return 0.0;
        }
        else
        {
            cluster_thermo pars({
                    thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                    hadron.MI, cMTh(hadron, thArgs), hadron.BI
                });
            double integral = cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
            return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
        }
    }

    double cSDensityCluster(const d &hadron, const thermo &thArgs)
    {
        cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, cMTh(hadron, thArgs), hadron.BI
            });
        double integral = cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }

    cluster::cluster(double mi, double di, double bi, double ni, double si):
        MI(mi), DI(di), BI(bi), NI(ni), SI(si)
    {}

    pi0::pi0():
        cluster(G_MI_PI0, G_DI_PI0, G_BI_PI0, G_NI_PI0, G_SI_PI0)
    {}

    pi::pi():
        cluster(G_MI_PI, G_DI_PI, G_BI_PI, G_NI_PI, G_SI_PI)
    {}

    K::K():
        cluster(G_MI_K, G_DI_K, G_BI_K, G_NI_K, G_SI_K)
    {}

    K0::K0():
        cluster(G_MI_K0, G_DI_K0, G_BI_K0, G_NI_K0, G_SI_K0)
    {}

    eta::eta():
        cluster(G_MI_ETA, G_DI_ETA, G_BI_ETA, G_NI_ETA, G_SI_ETA)
    {}

    rho::rho():
        cluster(G_MI_RHO, G_DI_RHO, G_BI_RHO, G_NI_RHO, G_SI_RHO)
    {}

    Kstar_892::Kstar_892():
        cluster(G_MI_KSTAR_892, G_DI_KSTAR_892, G_BI_KSTAR_892, G_NI_KSTAR_892, G_SI_KSTAR_892)
    {}

    omega::omega():
        cluster(G_MI_OMEGA, G_DI_OMEGA, G_BI_OMEGA, G_NI_OMEGA, G_SI_OMEGA)
    {}

    Kstar0_892::Kstar0_892():
        cluster(G_MI_KSTAR0_892, G_DI_KSTAR0_892, G_BI_KSTAR0_892, G_NI_KSTAR0_892, G_SI_KSTAR0_892)
    {}

    p::p():
        cluster(G_MI_P, G_DI_P, G_BI_P, G_NI_P, G_SI_P)
    {}

    n::n():
        cluster(G_MI_N, G_DI_N, G_BI_N, G_NI_N, G_SI_N)
    {}

    etaPrime::etaPrime():
        cluster(G_MI_ETAPRIME, G_DI_ETAPRIME, G_BI_ETAPRIME, G_NI_ETAPRIME, G_SI_ETAPRIME)
    {}

    a0::a0():
        cluster(G_MI_A0, G_DI_A0, G_BI_A0, G_NI_A0, G_SI_A0)
    {}

    f0::f0():
        cluster(G_MI_F0, G_DI_F0, G_BI_F0, G_NI_F0, G_SI_F0)
    {}

    phi::phi():
        cluster(G_MI_PHI, G_DI_PHI, G_BI_PHI, G_NI_PHI, G_SI_PHI)
    {}

    Lambda::Lambda():
        cluster(G_MI_LAMBDA, G_DI_LAMBDA, G_BI_LAMBDA, G_NI_LAMBDA, G_SI_LAMBDA)
    {}

    h1::h1():
        cluster(G_MI_H1, G_DI_H1, G_BI_H1, G_NI_H1, G_SI_H1)
    {}

    SigmaPlus::SigmaPlus():
        cluster(G_MI_SIGMAPLUS, G_DI_SIGMAPLUS, G_BI_SIGMAPLUS, G_NI_SIGMAPLUS, G_SI_SIGMAPLUS)
    {}

    Sigma0::Sigma0():
        cluster(G_MI_SIGMA0, G_DI_SIGMA0, G_BI_SIGMA0, G_NI_SIGMA0, G_SI_SIGMA0)
    {}

    SigmaMinus::SigmaMinus():
        cluster(G_MI_SIGMAMINUS, G_DI_SIGMAMINUS, G_BI_SIGMAMINUS, G_NI_SIGMAMINUS, G_SI_SIGMAMINUS)
    {}

    b1::b1():
        cluster(G_MI_B1, G_DI_B1, G_BI_B1, G_NI_B1, G_SI_B1)
    {}

    a1::a1():
        cluster(G_MI_A1, G_DI_A1, G_BI_A1, G_NI_A1, G_SI_A1)
    {}

    Delta::Delta():
        cluster(G_MI_DELTA, G_DI_DELTA, G_BI_DELTA, G_NI_DELTA, G_SI_DELTA)
    {}

    D1::D1():
        cluster(G_MI_D1, G_DI_D1, G_BI_D1, G_NI_D1, G_SI_D1)
    {}

    D2::D2():
        cluster(G_MI_D2, G_DI_D2, G_BI_D2, G_NI_D2, G_SI_D2)
    {}

    FourQ1::FourQ1():
        cluster(G_MI_4Q1, G_DI_4Q1, G_BI_4Q1, G_NI_4Q1, G_SI_4Q1)
    {}

    FourQ2::FourQ2():
        cluster(G_MI_4Q2, G_DI_4Q2, G_BI_4Q2, G_NI_4Q2, G_SI_4Q2)
    {}

    FourQ3::FourQ3():
        cluster(G_MI_4Q3, G_DI_4Q3, G_BI_4Q3, G_NI_4Q3, G_SI_4Q3)
    {}

    FourQ4::FourQ4():
        cluster(G_MI_4Q4, G_DI_4Q4, G_BI_4Q4, G_NI_4Q4, G_SI_4Q4)
    {}

    P1::P1():
        cluster(G_MI_P1, G_DI_P1, G_BI_P1, G_NI_P1, G_SI_P1)
    {}

    P2::P2():
        cluster(G_MI_P2, G_DI_P2, G_BI_P2, G_NI_P2, G_SI_P2)
    {}

    FiveQ1::FiveQ1():
        cluster(G_MI_5Q1, G_DI_5Q1, G_BI_5Q1, G_NI_5Q1, G_SI_5Q1)
    {}

    FiveQ2::FiveQ2():
        cluster(G_MI_5Q2, G_DI_5Q2, G_BI_5Q2, G_NI_5Q2, G_SI_5Q2)
    {}

    FiveQ3::FiveQ3():
        cluster(G_MI_5Q3, G_DI_5Q3, G_BI_5Q3, G_NI_5Q3, G_SI_5Q3)
    {}

    d::d():
        cluster(G_MI_D, G_DI_D, G_BI_D, G_NI_D, G_SI_D)
    {}


    Kstar0_892::~Kstar0_892() {}
    SigmaMinus::~SigmaMinus() {}
    Kstar_892::~Kstar_892() {}
    SigmaPlus::~SigmaPlus() {}
    etaPrime::~etaPrime() {}
    cluster::~cluster() {}
    Lambda::~Lambda() {}
    Sigma0::~Sigma0() {}
    FourQ1::~FourQ1() {}
    FourQ2::~FourQ2() {}
    FourQ3::~FourQ3() {}
    FourQ4::~FourQ4() {}
    FiveQ1::~FiveQ1() {}
    FiveQ2::~FiveQ2() {}
    FiveQ3::~FiveQ3() {}
    omega::~omega() {}
    Delta::~Delta() {}
    pi0::~pi0() {}
    eta::~eta() {}
    rho::~rho() {}
    phi::~phi() {}
    pi::~pi() {}
    K0::~K0() {}
    a0::~a0() {}
    f0::~f0() {}
    h1::~h1() {}
    b1::~b1() {}
    a1::~a1() {}
    D1::~D1() {}
    D2::~D2() {}
    P1::~P1() {}
    P2::~P2() {}
    K::~K() {}
    p::~p() {}
    n::~n() {}
    d::~d() {}

    static PyObject* cSDensityClusterAll(PyObject *self, PyObject *args)
    {
        Py_ssize_t lenT, lenMuB, lenPhiRe, lenPhiIm;
        PyObject *PyT, *PyMuB, *PyPhiRe, *PyPhiIm;

        if (! PyArg_ParseTuple(
                args, "O!O!O!O!", 
                &PyTuple_Type, &PyT,
                &PyTuple_Type, &PyMuB,
                &PyTuple_Type, &PyPhiRe,
                &PyTuple_Type, &PyPhiIm
                )) return NULL;

        lenPhiRe = PyTuple_Size(PyPhiRe);
        lenPhiIm = PyTuple_Size(PyPhiIm);
        lenMuB = PyTuple_Size(PyMuB);
        lenT = PyTuple_Size(PyT);
        
        if (lenT != lenMuB || lenT != lenPhiRe || lenT != lenPhiIm) return NULL;
        if (lenT < 0) return NULL;

        PyObject *PyTotal = PyTuple_New(lenT);
        PyObject *PyPartial = PyTuple_New(lenT);

        std::vector<double> sTotal(lenT), sPi0(lenT), sPi(lenT), sK(lenT), sK0(lenT);
        std::vector<double> sKStar892(lenT), sKStar0892(lenT), sP(lenT), sN(lenT);
        std::vector<double> sPhi(lenT), sLambda(lenT), sH1(lenT), sSigmaPlus(lenT);
        std::vector<double> sB1(lenT), sA1(lenT), sDelta(lenT), sD1(lenT), sD2(lenT);
        std::vector<double> sP1(lenT), sP2(lenT), s5Q1(lenT), s5Q2(lenT), s5Q3(lenT);
        std::vector<double> sEta(lenT), sRho(lenT), sOmega(lenT), sEtaPrime(lenT);
        std::vector<double> sF0(lenT), sSigma0(lenT), sSigmaMinus(lenT), s4Q3(lenT);
        std::vector<double> s4Q1(lenT), s4Q2(lenT), sD(lenT), sA0(lenT), s4Q4(lenT);

        pi0 pPi0; pi pPi; K pK; K0 pK0; eta pEta;
        Kstar_892 pKStar892; rho pRho; omega pOmega;
        Kstar0_892 pKStar0892; p pP; d pD; FiveQ3 p5Q3;
        n pN; etaPrime pEtaPrime; a0 pA0; f0 pF0;
        phi pPhi; Lambda pLambda; h1 pH1; SigmaPlus pSigmaPlus;
        Sigma0 pSigma0; SigmaMinus pSigmaMinus; b1 pB1;
        a1 pA1; Delta pDelta; D1 pD1; D2 pD2;
        FourQ1 p4Q1; FourQ2 p4Q2; FourQ3 p4Q3; FourQ4 p4Q4;
        P1 pP1; P2 pP2; FiveQ1 p5Q1; FiveQ2 p5Q2;

        for (Py_ssize_t i = 0; i < lenT; i++)
        {
            // pbar(i, 0, lenT);

            std::cout << i << std::endl;

            thermo tThArgs({
                PyFloat_AsDouble(PyTuple_GetItem(PyT, i)),
                PyFloat_AsDouble(PyTuple_GetItem(PyMuB, i)),
                PyFloat_AsDouble(PyTuple_GetItem(PyPhiRe, i)),
                PyFloat_AsDouble(PyTuple_GetItem(PyPhiIm, i))
            });

            double tTot = 0.0;
            // std::cout << "pi0" << std::endl;
            double tPi0 = cSDensityCluster(pPi0, tThArgs);
            // std::cout << "pi" << std::endl;
            double tPi = cSDensityCluster(pPi, tThArgs);
            // std::cout << "K" << std::endl;
            double tK = cSDensityCluster(pK, tThArgs);
            // std::cout << "K0" << std::endl;
            double tK0 = cSDensityCluster(pK0, tThArgs);
            // std::cout << "eta" << std::endl;
            double tEta = cSDensityCluster(pEta, tThArgs);
            // std::cout << "KStar892" << std::endl;
            double tKStar892 = cSDensityCluster(pKStar892, tThArgs);
            // std::cout << "rho" << std::endl;
            double tRho = cSDensityCluster(pRho, tThArgs);
            // std::cout << "omega" << std::endl;
            double tOmega = cSDensityCluster(pOmega, tThArgs);
            // std::cout << "KStar0892" << std::endl;
            double tKStar0892 = cSDensityCluster(pKStar0892, tThArgs);
            // std::cout << "p" << std::endl;
            double tP = cSDensityCluster(pP, tThArgs);
            // std::cout << "d" << std::endl;
            double tD = cSDensityCluster(pD, tThArgs);
            // std::cout << "5q3" << std::endl;
            double t5Q3 = cSDensityCluster(p5Q3, tThArgs);
            // std::cout << "n" << std::endl;
            double tN = cSDensityCluster(pN, tThArgs);
            // std::cout << "etaPrime" << std::endl;
            double tEtaPrime = cSDensityCluster(pEtaPrime, tThArgs);
            // std::cout << "a0" << std::endl;
            double tA0 = cSDensityCluster(pA0, tThArgs);
            // std::cout << "f0" << std::endl;
            double tF0 = cSDensityCluster(pF0, tThArgs);
            // std::cout << "phi" << std::endl;
            double tPhi = cSDensityCluster(pPhi, tThArgs);
            // std::cout << "Lambda" << std::endl;
            double tLambda = cSDensityCluster(pLambda, tThArgs);
            // std::cout << "h1" << std::endl;
            double tH1 = cSDensityCluster(pH1, tThArgs);
            // std::cout << "SigmaPlus" << std::endl;
            double tSigmaPlus = cSDensityCluster(pSigmaPlus, tThArgs);
            // std::cout << "Sigma0" << std::endl;
            double tSigma0 = cSDensityCluster(pSigma0, tThArgs);
            // std::cout << "SigmaMinus" << std::endl;
            double tSigmaMinus = cSDensityCluster(pSigmaMinus, tThArgs);
            // std::cout << "b1" << std::endl;
            double tB1 = cSDensityCluster(pB1, tThArgs);
            // std::cout << "a1" << std::endl;
            double tA1 = cSDensityCluster(pA1, tThArgs);
            // std::cout << "Delta" << std::endl;
            double tDelta = cSDensityCluster(pDelta, tThArgs);
            // std::cout << "D1" << std::endl;
            double tD1 = cSDensityCluster(pD1, tThArgs);
            // std::cout << "D2" << std::endl;
            double tD2 = cSDensityCluster(pD2, tThArgs);
            // std::cout << "4q1" << std::endl;
            double t4Q1 = cSDensityCluster(p4Q1, tThArgs);
            // std::cout << "4q2" << std::endl;
            double t4Q2 = cSDensityCluster(p4Q2, tThArgs);
            // std::cout << "4q3" << std::endl;
            double t4Q3 = cSDensityCluster(p4Q3, tThArgs);
            // std::cout << "4q4" << std::endl;
            double t4Q4 = cSDensityCluster(p4Q4, tThArgs);
            // std::cout << "P1" << std::endl;
            double tP1 = cSDensityCluster(pP1, tThArgs);
            // std::cout << "P2" << std::endl;
            double tP2 = cSDensityCluster(pP2, tThArgs);
            // std::cout << "5q1" << std::endl;
            double t5Q1 = cSDensityCluster(p5Q1, tThArgs);
            // std::cout << "5q2" << std::endl;
            double t5Q2 = cSDensityCluster(p5Q2, tThArgs);

            tTot += tPi0 + tPi + tK + tK0 + tEta + tKStar892 + tRho + tOmega + tKStar0892;
            tTot += tP + tD + t5Q3 + tN + tEtaPrime + tA0 + tF0 + tPhi + tLambda + tH1;
            tTot += tSigmaPlus + tSigma0 + tSigmaMinus + tB1 + tA1 + tDelta + tD1 + tD2;
            tTot += t4Q1 + t4Q2 + t4Q3 + t4Q4 + tP1 + tP2 + t5Q1 + t5Q2;

            PyTuple_SetItem(PyTotal, i, Py_BuildValue("d", tTot));
            PyTuple_SetItem(PyPartial, i, Py_BuildValue(
                "(ddddddddddddddddddddddddddddddddddd)", 
                tPi0, tPi, tK, tK0, tEta, tKStar892, tRho, tOmega, tKStar0892, tP, tD, t5Q3,
                tN, tEtaPrime, tA0, tF0, tPhi, tLambda, tH1, tSigmaPlus, tSigma0, tSigmaMinus,
                tB1, tA1, tDelta, tD1, tD2, t4Q1, t4Q2, t4Q3, t4Q4, tP1, tP2, t5Q1, t5Q2
            ));
        }
        PyObject *result = Py_BuildValue("(OO)", PyTotal, PyPartial);
        Py_DECREF(PyTotal);
        Py_DECREF(PyPartial);
        return result;
    }

    static PyObject* cBDensityClusterAll(PyObject *self, PyObject *args)
    {
        Py_ssize_t lenT, lenMuB, lenPhiRe, lenPhiIm;
        PyObject *PyT, *PyMuB, *PyPhiRe, *PyPhiIm;

        if (! PyArg_ParseTuple(
                args, "O!O!O!O!", 
                &PyTuple_Type, &PyT,
                &PyTuple_Type, &PyMuB,
                &PyTuple_Type, &PyPhiRe,
                &PyTuple_Type, &PyPhiIm
                )) return NULL;

        lenPhiRe = PyTuple_Size(PyPhiRe);
        lenPhiIm = PyTuple_Size(PyPhiIm);
        lenMuB = PyTuple_Size(PyMuB);
        lenT = PyTuple_Size(PyT);
        
        if (lenT != lenMuB || lenT != lenPhiRe || lenT != lenPhiIm) return NULL;
        if (lenT < 0) return NULL;

        PyObject *PyTotal = PyTuple_New(lenT);
        PyObject *PyPartial = PyTuple_New(lenT);

        std::vector<double> sTotal(lenT), sPi0(lenT), sPi(lenT), sK(lenT), sK0(lenT);
        std::vector<double> sKStar892(lenT), sKStar0892(lenT), sP(lenT), sN(lenT);
        std::vector<double> sPhi(lenT), sLambda(lenT), sH1(lenT), sSigmaPlus(lenT);
        std::vector<double> sB1(lenT), sA1(lenT), sDelta(lenT), sD1(lenT), sD2(lenT);
        std::vector<double> sP1(lenT), sP2(lenT), s5Q1(lenT), s5Q2(lenT), s5Q3(lenT);
        std::vector<double> sEta(lenT), sRho(lenT), sOmega(lenT), sEtaPrime(lenT);
        std::vector<double> sF0(lenT), sSigma0(lenT), sSigmaMinus(lenT), s4Q3(lenT);
        std::vector<double> s4Q1(lenT), s4Q2(lenT), sD(lenT), sA0(lenT), s4Q4(lenT);

        pi0 pPi0; pi pPi; K pK; K0 pK0; eta pEta;
        Kstar_892 pKStar892; rho pRho; omega pOmega;
        Kstar0_892 pKStar0892; p pP; d pD; FiveQ3 p5Q3;
        n pN; etaPrime pEtaPrime; a0 pA0; f0 pF0;
        phi pPhi; Lambda pLambda; h1 pH1; SigmaPlus pSigmaPlus;
        Sigma0 pSigma0; SigmaMinus pSigmaMinus; b1 pB1;
        a1 pA1; Delta pDelta; D1 pD1; D2 pD2;
        FourQ1 p4Q1; FourQ2 p4Q2; FourQ3 p4Q3; FourQ4 p4Q4;
        P1 pP1; P2 pP2; FiveQ1 p5Q1; FiveQ2 p5Q2;

        for (Py_ssize_t i = 0; i < lenT; i++)
        {
            // pbar(i, 0, lenT);

            std::cout << i << std::endl;

            thermo tThArgs({
                PyFloat_AsDouble(PyTuple_GetItem(PyT, i)),
                PyFloat_AsDouble(PyTuple_GetItem(PyMuB, i)),
                PyFloat_AsDouble(PyTuple_GetItem(PyPhiRe, i)),
                PyFloat_AsDouble(PyTuple_GetItem(PyPhiIm, i))
            });

            double tTot = 0.0;
            // std::cout << "pi0" << std::endl;
            double tPi0 = cBDensityCluster(pPi0, tThArgs);
            // std::cout << "pi" << std::endl;
            double tPi = cBDensityCluster(pPi, tThArgs);
            // std::cout << "K" << std::endl;
            double tK = cBDensityCluster(pK, tThArgs);
            // std::cout << "K0" << std::endl;
            double tK0 = cBDensityCluster(pK0, tThArgs);
            // std::cout << "eta" << std::endl;
            double tEta = cBDensityCluster(pEta, tThArgs);
            // std::cout << "KStar892" << std::endl;
            double tKStar892 = cBDensityCluster(pKStar892, tThArgs);
            // std::cout << "rho" << std::endl;
            double tRho = cBDensityCluster(pRho, tThArgs);
            // std::cout << "omega" << std::endl;
            double tOmega = cBDensityCluster(pOmega, tThArgs);
            // std::cout << "KStar0892" << std::endl;
            double tKStar0892 = cBDensityCluster(pKStar0892, tThArgs);
            // std::cout << "p" << std::endl;
            double tP = cBDensityCluster(pP, tThArgs);
            // std::cout << "d" << std::endl;
            double tD = cBDensityCluster(pD, tThArgs);
            // std::cout << "5q3" << std::endl;
            double t5Q3 = cBDensityCluster(p5Q3, tThArgs);
            // std::cout << "n" << std::endl;
            double tN = cBDensityCluster(pN, tThArgs);
            // std::cout << "etaPrime" << std::endl;
            double tEtaPrime = cBDensityCluster(pEtaPrime, tThArgs);
            // std::cout << "a0" << std::endl;
            double tA0 = cBDensityCluster(pA0, tThArgs);
            // std::cout << "f0" << std::endl;
            double tF0 = cBDensityCluster(pF0, tThArgs);
            // std::cout << "phi" << std::endl;
            double tPhi = cBDensityCluster(pPhi, tThArgs);
            // std::cout << "Lambda" << std::endl;
            double tLambda = cBDensityCluster(pLambda, tThArgs);
            // std::cout << "h1" << std::endl;
            double tH1 = cBDensityCluster(pH1, tThArgs);
            // std::cout << "SigmaPlus" << std::endl;
            double tSigmaPlus = cBDensityCluster(pSigmaPlus, tThArgs);
            // std::cout << "Sigma0" << std::endl;
            double tSigma0 = cBDensityCluster(pSigma0, tThArgs);
            // std::cout << "SigmaMinus" << std::endl;
            double tSigmaMinus = cBDensityCluster(pSigmaMinus, tThArgs);
            // std::cout << "b1" << std::endl;
            double tB1 = cBDensityCluster(pB1, tThArgs);
            // std::cout << "a1" << std::endl;
            double tA1 = cBDensityCluster(pA1, tThArgs);
            // std::cout << "Delta" << std::endl;
            double tDelta = cBDensityCluster(pDelta, tThArgs);
            // std::cout << "D1" << std::endl;
            double tD1 = cBDensityCluster(pD1, tThArgs);
            // std::cout << "D2" << std::endl;
            double tD2 = cBDensityCluster(pD2, tThArgs);
            // std::cout << "4q1" << std::endl;
            double t4Q1 = cBDensityCluster(p4Q1, tThArgs);
            // std::cout << "4q2" << std::endl;
            double t4Q2 = cBDensityCluster(p4Q2, tThArgs);
            // std::cout << "4q3" << std::endl;
            double t4Q3 = cBDensityCluster(p4Q3, tThArgs);
            // std::cout << "4q4" << std::endl;
            double t4Q4 = cBDensityCluster(p4Q4, tThArgs);
            // std::cout << "P1" << std::endl;
            double tP1 = cBDensityCluster(pP1, tThArgs);
            // std::cout << "P2" << std::endl;
            double tP2 = cBDensityCluster(pP2, tThArgs);
            // std::cout << "5q1" << std::endl;
            double t5Q1 = cBDensityCluster(p5Q1, tThArgs);
            // std::cout << "5q2" << std::endl;
            double t5Q2 = cBDensityCluster(p5Q2, tThArgs);

            tTot += tPi0 + tPi + tK + tK0 + tEta + tKStar892 + tRho + tOmega + tKStar0892;
            tTot += tP + tD + t5Q3 + tN + tEtaPrime + tA0 + tF0 + tPhi + tLambda + tH1;
            tTot += tSigmaPlus + tSigma0 + tSigmaMinus + tB1 + tA1 + tDelta + tD1 + tD2;
            tTot += t4Q1 + t4Q2 + t4Q3 + t4Q4 + tP1 + tP2 + t5Q1 + t5Q2;

            PyTuple_SetItem(PyTotal, i, Py_BuildValue("d", tTot));
            PyTuple_SetItem(PyPartial, i, Py_BuildValue(
                "(ddddddddddddddddddddddddddddddddddd)", 
                tPi0, tPi, tK, tK0, tEta, tKStar892, tRho, tOmega, tKStar0892, tP, tD, t5Q3,
                tN, tEtaPrime, tA0, tF0, tPhi, tLambda, tH1, tSigmaPlus, tSigma0, tSigmaMinus,
                tB1, tA1, tDelta, tD1, tD2, t4Q1, t4Q2, t4Q3, t4Q4, tP1, tP2, t5Q1, t5Q2
            ));
        }
        PyObject *result = Py_BuildValue("(OO)", PyTotal, PyPartial);
        Py_DECREF(PyTotal);
        Py_DECREF(PyPartial);
        return result;
    }

}

PyMODINIT_FUNC PyInit_cbscs(void) {
    return PyModule_Create(&bscs::module);
}