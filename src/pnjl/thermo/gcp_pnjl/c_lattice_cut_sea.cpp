#include "c_lattice_cut_sea.h"

gsl_function_wrapper cwPressureQLIntegrandReal(cPressureQLIntegrandReal);
gsl_function_wrapper cwPressureQSIntegrandReal(cPressureQSIntegrandReal);
gsl_function_wrapper cwBDensityQLIntegrandReal(cBDensityQLIntegrandReal);
gsl_function_wrapper cwBDensityQSIntegrandReal(cBDensityQSIntegrandReal);
gsl_function_wrapper cwSDensityQLIntegrandReal(cSDensityQLIntegrandReal);
gsl_function_wrapper cwSDensityQSIntegrandReal(cSDensityQSIntegrandReal);

double cTMott(const thermo &thArgs)
{
    return G_T_MOTT_0*(1.0 - G_KAPPA*pow(thArgs.muB/G_T_MOTT_0, 2));
}

double cTC(const thermo &thArgs)
{
    return G_T_C_0*(1.0 - G_KAPPA*pow(thArgs.muB/G_T_C_0, 2));
}

double cDeltaLS(const thermo &thArgs)
{
    return 0.5*(1 - tanh((thArgs.T - cTC(thArgs)) / G_DELTA_T));
}

double cPressureQLIntegrandReal(double p, void *pars)
{
    thermo thArgs = *(thermo *) pars;

    double mass = cML(thArgs);
    particle ppArgs({mass, 1.0/3.0}), paArgs({mass, -1.0/3.0});
    
    cdouble fp = cFFermionTriplet(p, ppArgs, thArgs);
    cdouble fm = cFFermionAntitriplet(p, paArgs, thArgs);
    
    return (pow(p, 4)/cEn(p, ppArgs))*(fp.real() + fm.real());
}

double cPressureQSIntegrandReal(double p, void *pars)
{
    thermo thArgs = *(thermo *) pars;

    double mass = cMS(thArgs);
    particle ppArgs({mass, 1.0/3.0}), paArgs({mass, -1.0/3.0});
    
    cdouble fp = cFFermionTriplet(p, ppArgs, thArgs);
    cdouble fm = cFFermionAntitriplet(p, paArgs, thArgs);
    
    return (pow(p, 4)/cEn(p, ppArgs))*(fp.real() + fm.real());
}

double cBDensityQLIntegrandReal(double p, void *pars)
{
    thermo thArgs = *(thermo *) pars;

    double mass = cML(thArgs);
    particle ppArgs({mass, 1.0/3.0}), paArgs({mass, -1.0/3.0});
    
    cdouble fp = cFFermionTriplet(p, ppArgs, thArgs);
    cdouble fm = cFFermionAntitriplet(p, paArgs, thArgs);
    
    return pow(p, 2)*(fp.real() - fm.real());
}

double cBDensityQSIntegrandReal(double p, void *pars)
{
    thermo thArgs = *(thermo *) pars;

    double mass = cMS(thArgs);
    particle ppArgs({mass, 1.0/3.0}), paArgs({mass, -1.0/3.0});
    
    cdouble fp = cFFermionTriplet(p, ppArgs, thArgs);
    cdouble fm = cFFermionAntitriplet(p, paArgs, thArgs);
    
    return pow(p, 2)*(fp.real() - fm.real());
}

double cSDensityQLIntegrandReal(double p, void *pars)
{
    thermo thArgs = *(thermo *) pars;

    double mass = cML(thArgs);
    particle ppArgs({mass, 1.0/3.0}), paArgs({mass, -1.0/3.0});
    
    cdouble fp = cFFermionTriplet(p, ppArgs, thArgs);
    cdouble fm = cFFermionAntitriplet(p, paArgs, thArgs);
    
    return pow(p, 2)*cEn(p, ppArgs)*(fp.real() + fm.real());
}

double cSDensityQSIntegrandReal(double p, void *pars)
{
    thermo thArgs = *(thermo *) pars;

    double mass = cMS(thArgs);
    particle ppArgs({mass, 1.0/3.0}), paArgs({mass, -1.0/3.0});
    
    cdouble fp = cFFermionTriplet(p, ppArgs, thArgs);
    cdouble fm = cFFermionAntitriplet(p, paArgs, thArgs);
    
    return pow(p, 2)*cEn(p, ppArgs)*(fp.real() + fm.real());
}

double cML(const thermo &thArgs)
{
    return thArgs.T <= cTMott(thArgs) ? G_SQRT2*G_M_L_VAC : G_M_0*cDeltaLS(thArgs) + G_M_C_L;
}

double cMS(const thermo &thArgs)
{
    return thArgs.T <= cTMott(thArgs) ? G_SQRT2*G_M_S_VAC : G_M_0*cDeltaLS(thArgs) + G_M_C_S;
}

double cPressureSea(const thermo &thArgs)
{
    return 0.0;
}

double cBDensitySea(const thermo &thArgs)
{
    return 0.0;
}

double cQNumberCumulantSea(const thermo &thArgs)
{
    return 0.0;
}

double cSDensitySea(const thermo &thArgs)
{
    return 0.0;
}

double cPressureField(const thermo &thArgs)
{
    return 0.0;
}

double cBDensityField(const thermo &thArgs)
{
    return 0.0;
}

double cQNumberCumulantField(const thermo &thArgs)
{
    return 0.0;
}

double cSDensityField(const thermo &thArgs)
{
    return 0.0;
}

double cPressureQ(char typ, thermo thArgs)
{
    double integral;
    if (typ == 'l')
    {
        integral = cwPressureQLIntegrandReal.qagiu(&thArgs, 0.0);
    }
    else if(typ == 's')
    {
        integral = cwPressureQSIntegrandReal.qagiu(&thArgs, 0.0);
    }
    else
    {
        throw std::invalid_argument("cPressureQ, invalid quark type");
    }
    return (G_NC/pow(M_PI, 2))*integral;
}

double cBDensityQ(char typ, thermo thArgs)
{
    if (thArgs.muB == 0.0)
    {
        return 0.0;
    }
    else
    {
        double integral;
        if (typ == 'l')
        {
            integral = cwBDensityQLIntegrandReal.qagiu(&thArgs, 0.0);
        }
        else if(typ == 's')
        {
            integral = cwBDensityQSIntegrandReal.qagiu(&thArgs, 0.0);
        }
        else
        {
            throw std::invalid_argument("cBDensityQ, invalid quark type");
        }
        return (G_NC/(3.0*pow(M_PI, 2)))*integral;
    }
}

double cQNumberCumulantQ(char typ, thermo thArgs)
{
    throw std::runtime_error("cQNumberCumulantQ not implemented yet");
    return 0.0;
}

double cSDensityQ(char typ, double p, double nB, thermo thArgs)
{
    if (thArgs.muB == 0.0)
    {
        double integral;
        if (typ == 'l')
        {
            integral = cwSDensityQLIntegrandReal.qagiu(&thArgs, 0.0);
        }
        else if(typ == 's')
        {
            integral = cwSDensityQSIntegrandReal.qagiu(&thArgs, 0.0);
        }
        else
        {
            throw std::invalid_argument("cSDensityQ, invalid quark type");
        }
        return (G_NC/((thArgs.T)*pow(M_PI, 2)))*integral + p;
    }
    else
    {
        double integral;
        if (typ == 'l')
        {
            integral = cwSDensityQLIntegrandReal.qagiu(&thArgs, 0.0);
        }
        else if(typ == 's')
        {
            integral = cwSDensityQSIntegrandReal.qagiu(&thArgs, 0.0);
        }
        else
        {
            throw std::invalid_argument("cSDensityQ, invalid quark type");
        }
        return (G_NC/(thArgs.T*pow(M_PI, 2)))*integral + p - ((thArgs.muB)*nB)/thArgs.T;
    }
}