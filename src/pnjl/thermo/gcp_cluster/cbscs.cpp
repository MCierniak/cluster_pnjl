#include "cbscs.h"

double bscs::cBDensityBosonSingletIntegrand(double p, void *pars)
{
    bscs::cluster_thermo pThArgs = *(bscs::cluster_thermo *) pars;
    thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
    double fpi = cFBosonSinglet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs);
    double fmi = cFBosonSinglet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs);
    double fpth = cFBosonSinglet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs);
    double fmth = cFBosonSinglet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs);
    return pow(p, 2)*(fpi - fmi - fpth + fmth);
}

double bscs::cBDensityFermionSingletIntegrand(double p, void *pars)
{
    bscs::cluster_thermo pThArgs = *(bscs::cluster_thermo *) pars;
    thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
    double fpi = cFFermionSinglet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs);
    double fmi = cFFermionSinglet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs);
    double fpth = cFFermionSinglet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs);
    double fmth = cFFermionSinglet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs);
    return pow(p, 2)*(fpi - fmi - fpth + fmth);
}

double bscs::cBDensityBosonTripletIntegrandReal(double p, void *pars)
{
    bscs::cluster_thermo pThArgs = *(bscs::cluster_thermo *) pars;
    thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
    double fpi = cFBosonTriplet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs).real();
    double fmi = cFBosonAntitriplet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs).real();
    double fpth = cFBosonTriplet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs).real();
    double fmth = cFBosonAntitriplet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs).real();
    return pow(p, 2)*(fpi - fmi - fpth + fmth);
}

double bscs::cBDensityBosonAntitripletIntegrandReal(double p, void *pars)
{
    bscs::cluster_thermo pThArgs = *(bscs::cluster_thermo *) pars;
    thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
    double fpi = cFBosonAntitriplet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs).real();
    double fmi = cFBosonTriplet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs).real();
    double fpth = cFBosonAntitriplet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs).real();
    double fmth = cFBosonTriplet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs).real();
    return pow(p, 2)*(fpi - fmi - fpth + fmth);
}

double bscs::cBDensityFermionTripletIntegrandReal(double p, void *pars)
{
    bscs::cluster_thermo pThArgs = *(bscs::cluster_thermo *) pars;
    thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
    double fpi = cFFermionTriplet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs).real();
    double fmi = cFFermionAntitriplet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs).real();
    double fpth = cFFermionTriplet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs).real();
    double fmth = cFFermionAntitriplet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs).real();
    return pow(p, 2)*(fpi - fmi - fpth + fmth);
}

double bscs::cBDensityFermionAntitripletIntegrandReal(double p, void *pars)
{
    bscs::cluster_thermo pThArgs = *(bscs::cluster_thermo *) pars;
    thermo thArgs = {pThArgs.T, pThArgs.muB, pThArgs.phiRe, pThArgs.phiIm};
    double fpi = cFFermionAntitriplet(p, particle({pThArgs.mi, pThArgs.chargeB}), thArgs).real();
    double fmi = cFFermionTriplet(p, particle({pThArgs.mi, -pThArgs.chargeB}), thArgs).real();
    double fpth = cFFermionAntitriplet(p, particle({pThArgs.mth, pThArgs.chargeB}), thArgs).real();
    double fmth = cFFermionTriplet(p, particle({pThArgs.mth, -pThArgs.chargeB}), thArgs).real();
    return pow(p, 2)*(fpi - fmi - fpth + fmth);
}

double bscs::cSDensityBosonSingletIntegrand(double p, void *pars)
{
    bscs::cluster_thermo pThArgs = *(bscs::cluster_thermo *) pars;
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

double bscs::cSDensityFermionSingletIntegrand(double p, void *pars)
{
    bscs::cluster_thermo pThArgs = *(bscs::cluster_thermo *) pars;
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

double bscs::cSDensityBosonTripletIntegrandReal(double p, void *pars)
{
    bscs::cluster_thermo pThArgs = *(bscs::cluster_thermo *) pars;
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

double bscs::cSDensityBosonAntitripletIntegrandReal(double p, void *pars)
{
    bscs::cluster_thermo pThArgs = *(bscs::cluster_thermo *) pars;
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

double bscs::cSDensityFermionTripletIntegrandReal(double p, void *pars)
{
    bscs::cluster_thermo pThArgs = *(bscs::cluster_thermo *) pars;
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

double bscs::cSDensityFermionAntitripletIntegrandReal(double p, void *pars)
{
    bscs::cluster_thermo pThArgs = *(bscs::cluster_thermo *) pars;
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

double bscs::cMTh(const bscs::cluster &hadron, const thermo &thArgs)
{
    return (hadron.NI - hadron.SI)*cML(thArgs) + hadron.SI*cMS(thArgs);
}

double bscs::cBDensityCluster(const bscs::pi0 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::pi0 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::pi &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::pi &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::K &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::K &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::K0 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::K0 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::eta &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::eta &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::rho &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::rho &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::omega &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::omega &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::Kstar_892 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::Kstar_892 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::Kstar0_892 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::Kstar0_892 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::p &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::p &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::n &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::n &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::etaPrime &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::etaPrime &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::a0 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::a0 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::f0 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::f0 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::phi &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::phi &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::Lambda &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::Lambda &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::h1 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::h1 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::SigmaPlus &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::SigmaPlus &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::Sigma0 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::Sigma0 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::SigmaMinus &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::SigmaMinus &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::b1 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::b1 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::a1 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::a1 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::Delta &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::Delta &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::D1 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonAntitripletIntegrandReal.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::D1 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonAntitripletIntegrandReal.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::D2 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonAntitripletIntegrandReal.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::D2 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonAntitripletIntegrandReal.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::FourQ1 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::FourQ1 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::FourQ2 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::FourQ2 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::FourQ3 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::FourQ3 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::FourQ4 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::FourQ4 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonTripletIntegrandReal.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::P1 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::P1 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::P2 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::P2 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityFermionSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::FiveQ1 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityFermionAntitripletIntegrandReal.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::FiveQ1 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityFermionAntitripletIntegrandReal.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::FiveQ2 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityFermionAntitripletIntegrandReal.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::FiveQ2 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityFermionAntitripletIntegrandReal.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::FiveQ3 &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityFermionAntitripletIntegrandReal.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::FiveQ3 &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityFermionAntitripletIntegrandReal.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

double bscs::cBDensityCluster(const bscs::d &hadron, const thermo &thArgs)
{
    if (hadron.BI == 0.0)
    {
        return 0.0;
    }
    else
    {
        bscs::cluster_thermo pars({
                thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
                hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
            });
        double integral = bscs::cwBDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
        return (hadron.BI*hadron.DI/(2.0*pow(M_PI, 2)))*integral;
    }
}

double bscs::cSDensityCluster(const bscs::d &hadron, const thermo &thArgs)
{
    bscs::cluster_thermo pars({
            thArgs.T, thArgs.muB, thArgs.phiRe, thArgs.phiIm,
            hadron.MI, bscs::cMTh(hadron, thArgs), hadron.BI
        });
    double integral = bscs::cwSDensityBosonSingletIntegrand.qagiu(&pars, 0.0);
    return -(hadron.DI/(2.0*pow(M_PI, 2)))*integral;
}

bscs::cluster::cluster(double mi, double di, double bi, double ni, double si):
    MI(mi), DI(di), BI(bi), NI(ni), SI(si)
{}

bscs::pi0::pi0():
    bscs::cluster(G_MI_PI0, G_DI_PI0, G_BI_PI0, G_NI_PI0, G_SI_PI0)
{}

bscs::pi::pi():
    bscs::cluster(G_MI_PI, G_DI_PI, G_BI_PI, G_NI_PI, G_SI_PI)
{}

bscs::K::K():
    bscs::cluster(G_MI_K, G_DI_K, G_BI_K, G_NI_K, G_SI_K)
{}

bscs::K0::K0():
    bscs::cluster(G_MI_K0, G_DI_K0, G_BI_K0, G_NI_K0, G_SI_K0)
{}

bscs::eta::eta():
    bscs::cluster(G_MI_ETA, G_DI_ETA, G_BI_ETA, G_NI_ETA, G_SI_ETA)
{}

bscs::rho::rho():
    bscs::cluster(G_MI_RHO, G_DI_RHO, G_BI_RHO, G_NI_RHO, G_SI_RHO)
{}

bscs::Kstar_892::Kstar_892():
    bscs::cluster(G_MI_KSTAR_892, G_DI_KSTAR_892, G_BI_KSTAR_892, G_NI_KSTAR_892, G_SI_KSTAR_892)
{}

bscs::omega::omega():
    bscs::cluster(G_MI_OMEGA, G_DI_OMEGA, G_BI_OMEGA, G_NI_OMEGA, G_SI_OMEGA)
{}

bscs::Kstar0_892::Kstar0_892():
    bscs::cluster(G_MI_KSTAR0_892, G_DI_KSTAR0_892, G_BI_KSTAR0_892, G_NI_KSTAR0_892, G_SI_KSTAR0_892)
{}

bscs::p::p():
    bscs::cluster(G_MI_P, G_DI_P, G_BI_P, G_NI_P, G_SI_P)
{}

bscs::n::n():
    bscs::cluster(G_MI_N, G_DI_N, G_BI_N, G_NI_N, G_SI_N)
{}

bscs::etaPrime::etaPrime():
    bscs::cluster(G_MI_ETAPRIME, G_DI_ETAPRIME, G_BI_ETAPRIME, G_NI_ETAPRIME, G_SI_ETAPRIME)
{}

bscs::a0::a0():
    bscs::cluster(G_MI_A0, G_DI_A0, G_BI_A0, G_NI_A0, G_SI_A0)
{}

bscs::f0::f0():
    bscs::cluster(G_MI_F0, G_DI_F0, G_BI_F0, G_NI_F0, G_SI_F0)
{}

bscs::phi::phi():
    bscs::cluster(G_MI_PHI, G_DI_PHI, G_BI_PHI, G_NI_PHI, G_SI_PHI)
{}

bscs::Lambda::Lambda():
    bscs::cluster(G_MI_LAMBDA, G_DI_LAMBDA, G_BI_LAMBDA, G_NI_LAMBDA, G_SI_LAMBDA)
{}

bscs::h1::h1():
    bscs::cluster(G_MI_H1, G_DI_H1, G_BI_H1, G_NI_H1, G_SI_H1)
{}

bscs::SigmaPlus::SigmaPlus():
    bscs::cluster(G_MI_SIGMAPLUS, G_DI_SIGMAPLUS, G_BI_SIGMAPLUS, G_NI_SIGMAPLUS, G_SI_SIGMAPLUS)
{}

bscs::Sigma0::Sigma0():
    bscs::cluster(G_MI_SIGMA0, G_DI_SIGMA0, G_BI_SIGMA0, G_NI_SIGMA0, G_SI_SIGMA0)
{}

bscs::SigmaMinus::SigmaMinus():
    bscs::cluster(G_MI_SIGMAMINUS, G_DI_SIGMAMINUS, G_BI_SIGMAMINUS, G_NI_SIGMAMINUS, G_SI_SIGMAMINUS)
{}

bscs::b1::b1():
    bscs::cluster(G_MI_B1, G_DI_B1, G_BI_B1, G_NI_B1, G_SI_B1)
{}

bscs::a1::a1():
    bscs::cluster(G_MI_A1, G_DI_A1, G_BI_A1, G_NI_A1, G_SI_A1)
{}

bscs::Delta::Delta():
    bscs::cluster(G_MI_DELTA, G_DI_DELTA, G_BI_DELTA, G_NI_DELTA, G_SI_DELTA)
{}

bscs::D1::D1():
    bscs::cluster(G_MI_D1, G_DI_D1, G_BI_D1, G_NI_D1, G_SI_D1)
{}

bscs::D2::D2():
    bscs::cluster(G_MI_D2, G_DI_D2, G_BI_D2, G_NI_D2, G_SI_D2)
{}

bscs::FourQ1::FourQ1():
    bscs::cluster(G_MI_4Q1, G_DI_4Q1, G_BI_4Q1, G_NI_4Q1, G_SI_4Q1)
{}

bscs::FourQ2::FourQ2():
    bscs::cluster(G_MI_4Q2, G_DI_4Q2, G_BI_4Q2, G_NI_4Q2, G_SI_4Q2)
{}

bscs::FourQ3::FourQ3():
    bscs::cluster(G_MI_4Q3, G_DI_4Q3, G_BI_4Q3, G_NI_4Q3, G_SI_4Q3)
{}

bscs::FourQ4::FourQ4():
    bscs::cluster(G_MI_4Q4, G_DI_4Q4, G_BI_4Q4, G_NI_4Q4, G_SI_4Q4)
{}

bscs::P1::P1():
    bscs::cluster(G_MI_P1, G_DI_P1, G_BI_P1, G_NI_P1, G_SI_P1)
{}

bscs::P2::P2():
    bscs::cluster(G_MI_P2, G_DI_P2, G_BI_P2, G_NI_P2, G_SI_P2)
{}

bscs::FiveQ1::FiveQ1():
    bscs::cluster(G_MI_5Q1, G_DI_5Q1, G_BI_5Q1, G_NI_5Q1, G_SI_5Q1)
{}

bscs::FiveQ2::FiveQ2():
    bscs::cluster(G_MI_5Q2, G_DI_5Q2, G_BI_5Q2, G_NI_5Q2, G_SI_5Q2)
{}

bscs::FiveQ3::FiveQ3():
    bscs::cluster(G_MI_5Q3, G_DI_5Q3, G_BI_5Q3, G_NI_5Q3, G_SI_5Q3)
{}

bscs::d::d():
    bscs::cluster(G_MI_D, G_DI_D, G_BI_D, G_NI_D, G_SI_D)
{}

bscs::Kstar0_892::~Kstar0_892() {}
bscs::SigmaMinus::~SigmaMinus() {}
bscs::Kstar_892::~Kstar_892() {}
bscs::SigmaPlus::~SigmaPlus() {}
bscs::etaPrime::~etaPrime() {}
bscs::Lambda::~Lambda() {}
bscs::Sigma0::~Sigma0() {}
bscs::FourQ1::~FourQ1() {}
bscs::FourQ2::~FourQ2() {}
bscs::FourQ3::~FourQ3() {}
bscs::FourQ4::~FourQ4() {}
bscs::FiveQ1::~FiveQ1() {}
bscs::FiveQ2::~FiveQ2() {}
bscs::FiveQ3::~FiveQ3() {}
bscs::omega::~omega() {}
bscs::Delta::~Delta() {}
bscs::pi0::~pi0() {}
bscs::eta::~eta() {}
bscs::rho::~rho() {}
bscs::phi::~phi() {}
bscs::pi::~pi() {}
bscs::K0::~K0() {}
bscs::a0::~a0() {}
bscs::f0::~f0() {}
bscs::h1::~h1() {}
bscs::b1::~b1() {}
bscs::a1::~a1() {}
bscs::D1::~D1() {}
bscs::D2::~D2() {}
bscs::P1::~P1() {}
bscs::P2::~P2() {}
bscs::K::~K() {}
bscs::p::~p() {}
bscs::n::~n() {}
bscs::d::~d() {}

static PyObject* bscs::cSDensityClusterAll(PyObject *self, PyObject *args)
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

    bscs::pi0 pPi0; bscs::pi pPi; bscs::K pK; bscs::K0 pK0; bscs::eta pEta;
    bscs::Kstar_892 pKStar892; bscs::rho pRho; bscs::omega pOmega;
    bscs::Kstar0_892 pKStar0892; bscs::p pP; bscs::d pD; bscs::FiveQ3 p5Q3;
    bscs::n pN; bscs::etaPrime pEtaPrime; bscs::a0 pA0; bscs::f0 pF0;
    bscs::phi pPhi; bscs::Lambda pLambda; bscs::h1 pH1; bscs::SigmaPlus pSigmaPlus;
    bscs::Sigma0 pSigma0; bscs::SigmaMinus pSigmaMinus; bscs::b1 pB1;
    bscs::a1 pA1; bscs::Delta pDelta; bscs::D1 pD1; bscs::D2 pD2;
    bscs::FourQ1 p4Q1; bscs::FourQ2 p4Q2; bscs::FourQ3 p4Q3; bscs::FourQ4 p4Q4;
    bscs::P1 pP1; bscs::P2 pP2; bscs::FiveQ1 p5Q1; bscs::FiveQ2 p5Q2;

    for (Py_ssize_t i = 0; i < lenT; i++)
    {
        pbar(i, 0, lenT);

        thermo tThArgs({
            PyFloat_AsDouble(PyTuple_GetItem(PyT, i)),
            PyFloat_AsDouble(PyTuple_GetItem(PyMuB, i)),
            PyFloat_AsDouble(PyTuple_GetItem(PyPhiRe, i)),
            PyFloat_AsDouble(PyTuple_GetItem(PyPhiIm, i))
        });

        double tTot = 0.0;
        double tPi0 = cSDensityCluster(pPi0, tThArgs);
        double tPi = cSDensityCluster(pPi, tThArgs);
        double tK = cSDensityCluster(pK, tThArgs);
        double tK0 = cSDensityCluster(pK0, tThArgs);
        double tEta = cSDensityCluster(pEta, tThArgs);
        double tKStar892 = cSDensityCluster(pKStar892, tThArgs);
        double tRho = cSDensityCluster(pRho, tThArgs);
        double tOmega = cSDensityCluster(pOmega, tThArgs);
        double tKStar0892 = cSDensityCluster(pKStar0892, tThArgs);
        double tP = cSDensityCluster(pP, tThArgs);
        double tD = cSDensityCluster(pD, tThArgs);
        double t5Q3 = cSDensityCluster(p5Q3, tThArgs);
        double tN = cSDensityCluster(pN, tThArgs);
        double tEtaPrime = cSDensityCluster(pEtaPrime, tThArgs);
        double tA0 = cSDensityCluster(pA0, tThArgs);
        double tF0 = cSDensityCluster(pF0, tThArgs);
        double tPhi = cSDensityCluster(pPhi, tThArgs);
        double tLambda = cSDensityCluster(pLambda, tThArgs);
        double tH1 = cSDensityCluster(pH1, tThArgs);
        double tSigmaPlus = cSDensityCluster(pSigmaPlus, tThArgs);
        double tSigma0 = cSDensityCluster(pSigma0, tThArgs);
        double tSigmaMinus = cSDensityCluster(pSigmaMinus, tThArgs);
        double tB1 = cSDensityCluster(pB1, tThArgs);
        double tA1 = cSDensityCluster(pA1, tThArgs);
        double tDelta = cSDensityCluster(pDelta, tThArgs);
        double tD1 = cSDensityCluster(pD1, tThArgs);
        double tD2 = cSDensityCluster(pD2, tThArgs);
        double t4Q1 = cSDensityCluster(p4Q1, tThArgs);
        double t4Q2 = cSDensityCluster(p4Q2, tThArgs);
        double t4Q3 = cSDensityCluster(p4Q3, tThArgs);
        double t4Q4 = cSDensityCluster(p4Q4, tThArgs);
        double tP1 = cSDensityCluster(pP1, tThArgs);
        double tP2 = cSDensityCluster(pP2, tThArgs);
        double t5Q1 = cSDensityCluster(p5Q1, tThArgs);
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

static PyObject* bscs::cBDensityClusterAll(PyObject *self, PyObject *args)
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

    bscs::pi0 pPi0; bscs::pi pPi; bscs::K pK; bscs::K0 pK0; bscs::eta pEta;
    bscs::Kstar_892 pKStar892; bscs::rho pRho; bscs::omega pOmega;
    bscs::Kstar0_892 pKStar0892; bscs::p pP; bscs::d pD; bscs::FiveQ3 p5Q3;
    bscs::n pN; bscs::etaPrime pEtaPrime; bscs::a0 pA0; bscs::f0 pF0;
    bscs::phi pPhi; bscs::Lambda pLambda; bscs::h1 pH1; bscs::SigmaPlus pSigmaPlus;
    bscs::Sigma0 pSigma0; bscs::SigmaMinus pSigmaMinus; bscs::b1 pB1;
    bscs::a1 pA1; bscs::Delta pDelta; bscs::D1 pD1; bscs::D2 pD2;
    bscs::FourQ1 p4Q1; bscs::FourQ2 p4Q2; bscs::FourQ3 p4Q3; bscs::FourQ4 p4Q4;
    bscs::P1 pP1; bscs::P2 pP2; bscs::FiveQ1 p5Q1; bscs::FiveQ2 p5Q2;

    for (Py_ssize_t i = 0; i < lenT; i++)
    {
        pbar(i, 0, lenT);

        thermo tThArgs({
            PyFloat_AsDouble(PyTuple_GetItem(PyT, i)),
            PyFloat_AsDouble(PyTuple_GetItem(PyMuB, i)),
            PyFloat_AsDouble(PyTuple_GetItem(PyPhiRe, i)),
            PyFloat_AsDouble(PyTuple_GetItem(PyPhiIm, i))
        });

        double tTot = 0.0;
        double tPi0 = cBDensityCluster(pPi0, tThArgs);
        double tPi = cBDensityCluster(pPi, tThArgs);
        double tK = cBDensityCluster(pK, tThArgs);
        double tK0 = cBDensityCluster(pK0, tThArgs);
        double tEta = cBDensityCluster(pEta, tThArgs);
        double tKStar892 = cBDensityCluster(pKStar892, tThArgs);
        double tRho = cBDensityCluster(pRho, tThArgs);
        double tOmega = cBDensityCluster(pOmega, tThArgs);
        double tKStar0892 = cBDensityCluster(pKStar0892, tThArgs);
        double tP = cBDensityCluster(pP, tThArgs);
        double tD = cBDensityCluster(pD, tThArgs);
        double t5Q3 = cBDensityCluster(p5Q3, tThArgs);
        double tN = cBDensityCluster(pN, tThArgs);
        double tEtaPrime = cBDensityCluster(pEtaPrime, tThArgs);
        double tA0 = cBDensityCluster(pA0, tThArgs);
        double tF0 = cBDensityCluster(pF0, tThArgs);
        double tPhi = cBDensityCluster(pPhi, tThArgs);
        double tLambda = cBDensityCluster(pLambda, tThArgs);
        double tH1 = cBDensityCluster(pH1, tThArgs);
        double tSigmaPlus = cBDensityCluster(pSigmaPlus, tThArgs);
        double tSigma0 = cBDensityCluster(pSigma0, tThArgs);
        double tSigmaMinus = cBDensityCluster(pSigmaMinus, tThArgs);
        double tB1 = cBDensityCluster(pB1, tThArgs);
        double tA1 = cBDensityCluster(pA1, tThArgs);
        double tDelta = cBDensityCluster(pDelta, tThArgs);
        double tD1 = cBDensityCluster(pD1, tThArgs);
        double tD2 = cBDensityCluster(pD2, tThArgs);
        double t4Q1 = cBDensityCluster(p4Q1, tThArgs);
        double t4Q2 = cBDensityCluster(p4Q2, tThArgs);
        double t4Q3 = cBDensityCluster(p4Q3, tThArgs);
        double t4Q4 = cBDensityCluster(p4Q4, tThArgs);
        double tP1 = cBDensityCluster(pP1, tThArgs);
        double tP2 = cBDensityCluster(pP2, tThArgs);
        double t5Q1 = cBDensityCluster(p5Q1, tThArgs);
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

PyMODINIT_FUNC PyInit_cbscs(void) {
    return PyModule_Create(&bscs::module);
}