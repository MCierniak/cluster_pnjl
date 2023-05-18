#include "c_distributions.h"

cdouble cZFermionTriplet_1(double logY, const thermo &thArgs)
{
    double ex1 = exp(-logY);
    double ex2 = exp(-2.0*logY);
    cdouble den(
        1.0 + 3.0*thArgs.phiRe*ex1 + 3.0*thArgs.phiRe*ex2 + exp(-3.0*logY),
        3.0*thArgs.phiIm*ex2 - 3.0*thArgs.phiIm*ex1
    );
    return log(den);
}

cdouble cZFermionTriplet_2(double logY, const thermo &thArgs)
{
    double ex1 = exp(-logY);
    cdouble den(
        exp(logY) + 3.0*thArgs.phiRe + 3.0*thArgs.phiRe*ex1 + exp(-2.0*logY),
        3.0*thArgs.phiIm*ex1 - 3.0*thArgs.phiIm
    );
    return log(den) - cdouble(logY, 0.0);
}

cdouble cZFermionTriplet_3(double logY, const thermo &thArgs)
{
    double ex1 = exp(-logY);
    cdouble den(
        3.0*thArgs.phiRe*exp(logY) + 3.0*thArgs.phiRe + ex1,
        3.0*thArgs.phiIm*exp(-2.0*logY) - 3.0*thArgs.phiIm*ex1
    );
    return log(den) - cdouble(2.0*logY, 0.0);
}

cdouble cZFermionAntitriplet_1(double logY, const thermo &thArgs)
{
    double ex1 = exp(-logY);
    double ex2 = exp(-2.0*logY);
    cdouble den(
        1.0 + 3.0*thArgs.phiRe*ex1 + 3.0*thArgs.phiRe*ex2 + exp(-3.0*logY),
        3.0*(-thArgs.phiIm)*ex2 - 3.0*(-thArgs.phiIm)*ex1
    );
    return log(den);
}

cdouble cZFermionAntitriplet_2(double logY, const thermo &thArgs)
{
    double ex1 = exp(-logY);
    cdouble den(
        exp(logY) + 3.0*thArgs.phiRe + 3.0*thArgs.phiRe*ex1 + exp(-2.0*logY),
        3.0*(-thArgs.phiIm)*ex1 - 3.0*(-thArgs.phiIm)
    );
    return log(den) - cdouble(logY, 0.0);
}

cdouble cZFermionAntitriplet_3(double logY, const thermo &thArgs)
{
    double ex1 = exp(-logY);
    cdouble den(
        3.0*thArgs.phiRe*exp(logY) + 3.0*thArgs.phiRe + ex1,
        3.0*(-thArgs.phiIm)*exp(-2.0*logY) - 3.0*(-thArgs.phiIm)*ex1
    );
    return log(den) - cdouble(2.0*logY, 0.0);
}

cdouble cZBosonTriplet_1(double logY, const thermo &thArgs)
{
    double ex1 = exp(-logY);
    double ex2 = exp(-2.0*logY);
    cdouble den(
        1.0 - 3.0*thArgs.phiRe*ex1 + 3.0*thArgs.phiRe*ex2 - exp(-3.0*logY),
        3.0*thArgs.phiIm*ex1 + 3.0*thArgs.phiIm*ex2
    );
    return log(den);
}

cdouble cZBosonTriplet_2(double logY, const thermo &thArgs)
{
    double ex1 = exp(-logY);
    cdouble den(
        exp(logY) - 3.0*thArgs.phiRe + 3.0*thArgs.phiRe*ex1 - exp(-2.0*logY),
        3.0*thArgs.phiIm + 3.0*thArgs.phiIm*ex1
    );
    return log(den) - cdouble(logY, 0.0);
}

cdouble cZBosonTriplet_3(double logY, const thermo &thArgs)
{
    double ex1 = exp(logY);
    cdouble den(
        exp(2.0*logY) - 3.0*thArgs.phiRe*ex1 + 3.0*thArgs.phiRe - exp(-logY),
        3.0*thArgs.phiIm*ex1 + 3.0*thArgs.phiIm
    );
    return log(den) - cdouble(2.0*logY, 0.0);
}

cdouble cZBosonAntitriplet_1(double logY, const thermo &thArgs)
{
    double ex1 = exp(-logY);
    double ex2 = exp(-2.0*logY);
    cdouble den(
        1.0 - 3.0*thArgs.phiRe*ex1 + 3.0*thArgs.phiRe*ex2 - exp(-3.0*logY),
        3.0*(-thArgs.phiIm)*ex1 + 3.0*(-thArgs.phiIm)*ex2
    );
    return log(den);
}

cdouble cZBosonAntitriplet_2(double logY, const thermo &thArgs)
{
    double ex1 = exp(-logY);
    cdouble den(
        exp(logY) - 3.0*thArgs.phiRe + 3.0*thArgs.phiRe*ex1 - exp(-2.0*logY),
        3.0*(-thArgs.phiIm) + 3.0*(-thArgs.phiIm)*ex1
    );
    return log(den) - cdouble(logY, 0.0);
}

cdouble cZBosonAntitriplet_3(double logY, const thermo &thArgs)
{
    double ex1 = exp(logY);
    cdouble den(
        exp(2.0*logY) - 3.0*thArgs.phiRe*ex1 + 3.0*thArgs.phiRe - exp(-logY),
        3.0*(-thArgs.phiIm)*ex1 + 3.0*(-thArgs.phiIm)
    );
    return log(den) - cdouble(2.0*logY, 0.0);
}

cdouble cFFermionTriplet_1(double logY, const thermo &thArgs)
{
    cdouble  numerator(
        thArgs.phiRe * exp(-logY) + 2.0 * thArgs.phiRe * exp(-2.0*logY) + exp(-3.0*logY),
        2.0 * thArgs.phiIm * exp(-2.0*logY) - thArgs.phiIm * exp(-logY)
    );
    cdouble  denominator(
        1.0 + 3.0 * thArgs.phiRe * exp(-logY) + 3.0 * thArgs.phiRe * exp(-2.0*logY) + exp(-3.0*logY),
        3.0 * thArgs.phiIm * exp(-2.0*logY) - 3.0 * thArgs.phiIm * exp(-logY)
    );
    return numerator / denominator;
}

cdouble cFFermionTriplet_2(double logY, const thermo &thArgs)
{
    cdouble  numerator(
        thArgs.phiRe * exp(logY) + 2.0 * thArgs.phiRe + exp(-logY),
        2.0 * thArgs.phiIm - thArgs.phiIm * exp(logY)
    );
    cdouble  denominator(
        exp(2.0*logY) + 3.0 * thArgs.phiRe * exp(logY) + 3.0 * thArgs.phiRe + exp(-logY),
        3.0 * thArgs.phiIm - 3.0 * thArgs.phiIm * exp(logY)
    );
    return numerator / denominator;
}

cdouble cFFermionAntitriplet_1(double logY, const thermo &thArgs)
{
    cdouble  numerator(
        thArgs.phiRe * exp(-logY) + 2.0 * thArgs.phiRe * exp(-2.0*logY) + exp(-3.0*logY),
        2.0 * (-thArgs.phiIm) * exp(-2.0*logY) - (-thArgs.phiIm) * exp(-logY)
    );
    cdouble  denominator(
        1.0 + 3.0 * thArgs.phiRe * exp(-logY) + 3.0 * thArgs.phiRe * exp(-2.0*logY) + exp(-3.0*logY),
        3.0 * (-thArgs.phiIm) * exp(-2.0*logY) - 3.0 * (-thArgs.phiIm) * exp(-logY)
    );
    return numerator / denominator;
}

cdouble cFFermionAntitriplet_2(double logY, const thermo &thArgs)
{
    cdouble  numerator(
        thArgs.phiRe * exp(logY) + 2.0 * thArgs.phiRe + exp(-logY),
        2.0 * (-thArgs.phiIm) - (-thArgs.phiIm) * exp(logY)
    );
    cdouble  denominator(
        exp(2.0*logY) + 3.0 * thArgs.phiRe * exp(logY) + 3.0 * thArgs.phiRe + exp(-logY),
        3.0 * (-thArgs.phiIm) - 3.0 * (-thArgs.phiIm) * exp(logY)
    );
    return numerator / denominator;
}

cdouble cFBosonTriplet_1(double logY, const thermo &thArgs)
{
    cdouble numerator(
        thArgs.phiRe * exp(-2.0*logY) + thArgs.phiRe * exp(-logY) * expm1(-logY) - exp(-3.0*logY),
        thArgs.phiIm * exp(-logY) + 2.0 * thArgs.phiIm * exp(-2.0*logY)
    );
    cdouble denominator(
        3.0 * thArgs.phiRe * exp(-logY) * expm1(-logY) + expm1(-3.0*logY),
        3.0 * thArgs.phiIm * exp(-logY) + 3.0 * thArgs.phiIm * exp(-2.0*logY)
    );
    return numerator / denominator;
}

cdouble cFBosonTriplet_2(double logY, const thermo &thArgs)
{
    cdouble numerator(
        thArgs.phiRe - thArgs.phiRe * expm1(logY) - exp(-logY),
        2.0 * thArgs.phiIm + thArgs.phiIm * exp(logY)
    );
    cdouble denominator(
        exp(2.0*logY) + exp(-logY) - 3.0 * thArgs.phiRe * expm1(logY),
        3.0 * thArgs.phiIm * exp(logY) + 3.0 * thArgs.phiIm
    );
    return numerator / denominator;
}

cdouble cFBosonAntitriplet_1(double logY, const thermo &thArgs)
{
    cdouble numerator(
        thArgs.phiRe * exp(-2.0*logY) + thArgs.phiRe * exp(-logY) * expm1(-logY) - exp(-3.0*logY),
        (-thArgs.phiIm) * exp(-logY) + 2.0 * (-thArgs.phiIm) * exp(-2.0*logY)
    );
    cdouble denominator(
        3.0 * thArgs.phiRe * exp(-logY) * expm1(-logY) + expm1(-3.0*logY),
        3.0 * (-thArgs.phiIm) * exp(-logY) + 3.0 * (-thArgs.phiIm) * exp(-2.0*logY)
    );
    return numerator / denominator;
}

cdouble cFBosonAntitriplet_2(double logY, const thermo &thArgs)
{
    cdouble numerator(
        thArgs.phiRe - thArgs.phiRe * expm1(logY) - exp(-logY),
        2.0 * (-thArgs.phiIm) + (-thArgs.phiIm) * exp(logY)
    );
    cdouble denominator(
        exp(2.0*logY) + exp(-logY) - 3.0 * thArgs.phiRe * expm1(logY),
        3.0 * (-thArgs.phiIm) * exp(logY) + 3.0 * (-thArgs.phiIm)
    );
    return numerator / denominator;
}

cdouble cdFdMBosonTriplet_aux(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    double log2 = 2.0*log1;
    if (log1 >= G_EXP_LIMIT)
    {
        return cdouble(0.0, 0.0);
    }
    else if (log1 < G_EXP_LIMIT && log2 >= G_EXP_LIMIT)
    {
        cdouble num1(
            2.0*(exp(-log1)*thArgs.phiRe - thArgs.phiRe),
            2.0*(exp(-log1)*thArgs.phiIm + thArgs.phiIm)
        );
        cdouble den1(
            exp(log1) + exp(-log1)*thArgs.phiRe - 2.0*thArgs.phiRe,
            exp(-log1)*thArgs.phiIm + 2.0*thArgs.phiIm
        );
        cdouble num2(
            2.0*(thArgs.phiRe - exp(-log1)),
            2.0*thArgs.phiIm
        );
        cdouble den2(
            exp(-log1) - 2.0*thArgs.phiRe + exp(log1)*thArgs.phiRe,
            -2.0*thArgs.phiIm - exp(log1)*thArgs.phiIm
        );
        return (num1/den1) + (num2/den2) - cdouble(1.0, 0.0);
    }
    else
    {
        cdouble num1(
            2.0*(thArgs.phiRe - exp(log1)*thArgs.phiRe),
            2.0*(thArgs.phiIm + exp(log1)*thArgs.phiIm)
        );
        cdouble den1(
            exp(log2) + thArgs.phiRe - 2.0*thArgs.phiRe*exp(log1),
            thArgs.phiIm + 2.0*thArgs.phiIm*exp(log1)
        );
        cdouble num2(
            2.0*(exp(log1)*thArgs.phiRe - 1.0),
            2.0*exp(log1)*thArgs.phiIm
        );
        cdouble den2(
            1.0 - 2.0*exp(log1)*thArgs.phiRe + exp(log2)*thArgs.phiRe,
            -2.0*exp(log1)*thArgs.phiIm - exp(log2)*thArgs.phiIm
        );
        return (num1/den1) + (num2/den2) - cdouble(1.0, 0.0);
    }
}

cdouble cdFdMBosonAntitriplet_aux(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    double log2 = 2.0*log1;
    if (log1 >= G_EXP_LIMIT)
    {
        return cdouble(0.0, 0.0);
    }
    else if (log1 < G_EXP_LIMIT && log2 >= G_EXP_LIMIT)
    {
        cdouble num1(
            2.0*(exp(-log1)*thArgs.phiRe - thArgs.phiRe),
            2.0*(exp(-log1)*(-thArgs.phiIm) + (-thArgs.phiIm))
        );
        cdouble den1(
            exp(log1) + exp(-log1)*thArgs.phiRe - 2.0*thArgs.phiRe,
            exp(-log1)*(-thArgs.phiIm) + 2.0*(-thArgs.phiIm)
        );
        cdouble num2(
            2.0*(thArgs.phiRe - exp(-log1)),
            2.0*(-thArgs.phiIm)
        );
        cdouble den2(
            exp(-log1) - 2.0*thArgs.phiRe + exp(log1)*thArgs.phiRe,
            -2.0*(-thArgs.phiIm) - exp(log1)*(-thArgs.phiIm)
        );
        return (num1/den1) + (num2/den2) - cdouble(1.0, 0.0);
    }
    else
    {
        cdouble num1(
            2.0*(thArgs.phiRe - exp(log1)*thArgs.phiRe),
            2.0*((-thArgs.phiIm) + exp(log1)*(-thArgs.phiIm))
        );
        cdouble den1(
            exp(log2) + thArgs.phiRe - 2.0*thArgs.phiRe*exp(log1),
            (-thArgs.phiIm) + 2.0*(-thArgs.phiIm)*exp(log1)
        );
        cdouble num2(
            2.0*(exp(log1)*thArgs.phiRe - 1.0),
            2.0*exp(log1)*(-thArgs.phiIm)
        );
        cdouble den2(
            1.0 - 2.0*exp(log1)*thArgs.phiRe + exp(log2)*thArgs.phiRe,
            -2.0*exp(log1)*(-thArgs.phiIm) - exp(log2)*(-thArgs.phiIm)
        );
        return (num1/den1) + (num2/den2) - cdouble(1.0, 0.0);
    }
}

cdouble cdFdMFermionTriplet_aux(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    double log2 = 2.0*log1;
    if (log1 >= G_EXP_LIMIT)
    {
        return cdouble(0.0, 0.0);
    }
    else if (log1 < G_EXP_LIMIT && log2 >= G_EXP_LIMIT)
    {
        cdouble num1(
            2.0*(exp(-log1)*thArgs.phiRe + thArgs.phiRe),
            2.0*(exp(-log1)*thArgs.phiIm - thArgs.phiIm)
        );
        cdouble den1(
            exp(log1) + exp(-log1)*thArgs.phiRe + 2.0*thArgs.phiRe,
            exp(-log1)*thArgs.phiIm - 2.0*thArgs.phiIm
        );
        cdouble num2(
            2.0*(exp(-log1) + thArgs.phiRe),
            2.0*thArgs.phiIm
        );
        cdouble den2(
            exp(-log1) + 2.0*thArgs.phiRe + exp(log1)*thArgs.phiRe,
            2.0*thArgs.phiIm - exp(log1)*thArgs.phiIm
        );
        return -(num1/den1) + (num2/den2) + cdouble(1.0, 0.0);
    }
    else
    {
        cdouble num1(
            2.0*(thArgs.phiRe + exp(log1)*thArgs.phiRe),
            2.0*(thArgs.phiIm - exp(log1)*thArgs.phiIm)
        );
        cdouble den1(
            exp(log2) + thArgs.phiRe + 2.0*thArgs.phiRe*exp(log1),
            thArgs.phiIm - 2.0*thArgs.phiIm*exp(log1)
        );
        cdouble num2(
            2.0*(1.0 + exp(log1)*thArgs.phiRe),
            2.0*exp(log1)*thArgs.phiIm
        );
        cdouble den2(
            1.0 + 2.0*exp(log1)*thArgs.phiRe + exp(log2)*thArgs.phiRe,
            2.0*exp(log1)*thArgs.phiIm - exp(log2)*thArgs.phiIm
        );
        return -(num1/den1) + (num2/den2) + cdouble(1.0, 0.0);
    }
}

cdouble cdFdMFermionAntitriplet_aux(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    double log2 = 2.0*log1;
    if (log1 >= G_EXP_LIMIT)
    {
        return cdouble(0.0, 0.0);
    }
    else if (log1 < G_EXP_LIMIT && log2 >= G_EXP_LIMIT)
    {
        cdouble num1(
            2.0*(exp(-log1)*thArgs.phiRe + thArgs.phiRe),
            2.0*(exp(-log1)*(-thArgs.phiIm) - (-thArgs.phiIm))
        );
        cdouble den1(
            exp(log1) + exp(-log1)*thArgs.phiRe + 2.0*thArgs.phiRe,
            exp(-log1)*(-thArgs.phiIm) - 2.0*(-thArgs.phiIm)
        );
        cdouble num2(
            2.0*(exp(-log1) + thArgs.phiRe),
            2.0*(-thArgs.phiIm)
        );
        cdouble den2(
            exp(-log1) + 2.0*thArgs.phiRe + exp(log1)*thArgs.phiRe,
            2.0*(-thArgs.phiIm) - exp(log1)*(-thArgs.phiIm)
        );
        return -(num1/den1) + (num2/den2) + cdouble(1.0, 0.0);
    }
    else
    {
        cdouble num1(
            2.0*(thArgs.phiRe + exp(log1)*thArgs.phiRe),
            2.0*((-thArgs.phiIm) - exp(log1)*(-thArgs.phiIm))
        );
        cdouble den1(
            exp(log2) + thArgs.phiRe + 2.0*thArgs.phiRe*exp(log1),
            (-thArgs.phiIm) - 2.0*(-thArgs.phiIm)*exp(log1)
        );
        cdouble num2(
            2.0*(1.0 + exp(log1)*thArgs.phiRe),
            2.0*exp(log1)*(-thArgs.phiIm)
        );
        cdouble den2(
            1.0 + 2.0*exp(log1)*thArgs.phiRe + exp(log2)*thArgs.phiRe,
            2.0*exp(log1)*(-thArgs.phiIm) - exp(log2)*(-thArgs.phiIm)
        );
        return -(num1/den1) + (num2/den2) + cdouble(1.0, 0.0);
    }
}

double cEn(double p, const particle &pArgs)
{
    return sqrt(std::pow(p, 2) + std::pow(pArgs.mass, 2));
}

double cLogY(double p, const particle &pArgs, const thermo &thArgs)
{
    return (cEn(p, pArgs) - pArgs.chargeB*thArgs.muB)/thArgs.T;
}

cdouble cZFermionTriplet(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    if (log1 >= G_EXP_LIMIT)
    {
        return -cdouble(3.0*log1, 0.0);
    }
    else if (log1 < G_EXP_LIMIT && 2.0*log1 >= G_EXP_LIMIT)
    {
        return cZFermionTriplet_3(log1, thArgs);
    }
    else if (log1 < G_EXP_LIMIT && 2.0*log1 < G_EXP_LIMIT && 3.0*log1 >= G_EXP_LIMIT)
    {
        return cZFermionTriplet_2(log1, thArgs);
    }
    else
    {
        return cZFermionTriplet_1(log1, thArgs);
    }
}

cdouble cZFermionAntitriplet(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    if (log1 >= G_EXP_LIMIT)
    {
        return -cdouble(3.0*log1, 0.0);
    }
    else if (log1 < G_EXP_LIMIT && 2.0*log1 >= G_EXP_LIMIT)
    {
        return cZFermionAntitriplet_3(log1, thArgs);
    }
    else if (log1 < G_EXP_LIMIT && 2.0*log1 < G_EXP_LIMIT && 3.0*log1 >= G_EXP_LIMIT)
    {
        return cZFermionAntitriplet_2(log1, thArgs);
    }
    else
    {
        return cZFermionAntitriplet_1(log1, thArgs);
    }
}

cdouble cZBosonTriplet(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    if (log1 >= G_EXP_LIMIT)
    {
        return -cdouble(3.0*log1, 0.0);
    }
    else if (log1 < G_EXP_LIMIT && 2.0*log1 >= G_EXP_LIMIT)
    {
        return cZBosonTriplet_3(log1, thArgs);
    }
    else if (log1 < G_EXP_LIMIT && 2.0*log1 < G_EXP_LIMIT && 3.0*log1 >= G_EXP_LIMIT)
    {
        return cZBosonTriplet_2(log1, thArgs);
    }
    else
    {
        return cZBosonTriplet_1(log1, thArgs);
    }
}

cdouble cZBosonAntitriplet(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    if (log1 >= G_EXP_LIMIT)
    {
        return -cdouble(3.0*log1, 0.0);
    }
    else if (log1 < G_EXP_LIMIT && 2.0*log1 >= G_EXP_LIMIT)
    {
        return cZBosonAntitriplet_3(log1, thArgs);
    }
    else if (log1 < G_EXP_LIMIT && 2.0*log1 < G_EXP_LIMIT && 3.0*log1 >= G_EXP_LIMIT)
    {
        return cZBosonAntitriplet_2(log1, thArgs);
    }
    else
    {
        return cZBosonAntitriplet_1(log1, thArgs);
    }
}

double cFFermionSinglet(double p, const particle &pArgs, const thermo &thArgs)
{
    double logY = cLogY(p, pArgs, thArgs);
    return logY >= G_EXP_LIMIT ? 0.0 : 1.0 / (exp(logY) + 1.0);
}

double cFBosonSinglet(double p, const particle &pArgs, const thermo &thArgs)
{
    double logY = cLogY(p, pArgs, thArgs);
    return logY >= G_EXP_LIMIT ? 0.0 : 1.0 / expm1(logY);
}

cdouble cFFermionTriplet(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    if(log1 <= -G_EXP_LIMIT)
    {
        return cdouble(1.0, 0.0);
    }
    else if (log1 > -G_EXP_LIMIT && 3.0*log1 <= -G_EXP_LIMIT)
    {
        return cFFermionTriplet_2(log1, thArgs);
    }
    else
    {
        return cFFermionTriplet_1(log1, thArgs);
    }
}

cdouble cFFermionAntitriplet(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    if(log1 <= -G_EXP_LIMIT)
    {
        return cdouble(1.0, 0.0);
    }
    else if (log1 > -G_EXP_LIMIT && 3.0*log1 <= -G_EXP_LIMIT)
    {
        return cFFermionAntitriplet_2(log1, thArgs);
    }
    else
    {
        return cFFermionAntitriplet_1(log1, thArgs);
    }
}

cdouble cFBosonTriplet(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    if(log1 <= -G_EXP_LIMIT)
    {
        return cdouble(-1.0, 0.0);
    }
    else if (log1 > -G_EXP_LIMIT && 3.0*log1 <= -G_EXP_LIMIT)
    {
        return cFBosonTriplet_2(log1, thArgs);
    }
    else
    {
        return cFBosonTriplet_1(log1, thArgs);
    }
}

cdouble cFBosonAntitriplet(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    if(log1 <= -G_EXP_LIMIT)
    {
        return cdouble(-1.0, 0.0);
    }
    else if (log1 > -G_EXP_LIMIT && 3.0*log1 <= -G_EXP_LIMIT)
    {
        return cFBosonAntitriplet_2(log1, thArgs);
    }
    else
    {
        return cFBosonAntitriplet_1(log1, thArgs);
    }
}

double cdFdMBosonSinglet(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    if(log1 >= G_EXP_LIMIT)
    {
        return 0.0;
    }
    else
    {
        return exp(log1) * pow(cFBosonSinglet(p, pArgs, thArgs), 2) * pArgs.mass / (cEn(p, pArgs) * thArgs.T);
    }
}

double cdFdMFermionSinglet(double p, const particle &pArgs, const thermo &thArgs)
{
    double log1 = cLogY(p, pArgs, thArgs);
    if (log1 >= G_EXP_LIMIT)
    {
        return 0.0;
    }
    else
    {
        return exp(log1) * pow(cFFermionSinglet(p, pArgs, thArgs), 2) * pArgs.mass / (cEn(p, pArgs) * thArgs.T);
    }
}

cdouble cdFdMBosonTriplet(double p, const particle &pArgs, const thermo &thArgs)
{
    cdouble fp = cFBosonTriplet(p, pArgs, thArgs);
    return (pow(fp, 2)+fp) * cdFdMBosonTriplet_aux(p, pArgs, thArgs) * pArgs.mass / (cEn(p, pArgs) * thArgs.T);
}

cdouble cdFdMBosonAntitriplet(double p, const particle &pArgs, const thermo &thArgs)
{
    cdouble fp = cFBosonAntitriplet(p, pArgs, thArgs);
    return (pow(fp, 2)+fp) * cdFdMBosonAntitriplet_aux(p, pArgs, thArgs) * pArgs.mass / (cEn(p, pArgs) * thArgs.T);
}

cdouble cdFdMFermionTriplet(double p, const particle &pArgs, const thermo &thArgs)
{
    cdouble fp = cFFermionTriplet(p, pArgs, thArgs);
    return (pow(fp, 2)-fp) * cdFdMFermionTriplet_aux(p, pArgs, thArgs) * pArgs.mass / (cEn(p, pArgs)*thArgs.T);
}

cdouble cdFdMFermionAntitriplet(double p, const particle &pArgs, const thermo &thArgs)
{
    cdouble fp = cFFermionAntitriplet(p, pArgs, thArgs);
    return (pow(fp, 2)-fp) * cdFdMFermionAntitriplet_aux(p, pArgs, thArgs) * pArgs.mass / (cEn(p, pArgs)*thArgs.T);
}
