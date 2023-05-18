#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include "../c_globals.h"

//Internal methods
cdouble cZFermionTriplet_1(double logY, const thermo &thArgs);
cdouble cZFermionTriplet_2(double logY, const thermo &thArgs);
cdouble cZFermionTriplet_3(double logY, const thermo &thArgs);
cdouble cZFermionAntitriplet_1(double logY, const thermo &thArgs);
cdouble cZFermionAntitriplet_2(double logY, const thermo &thArgs);
cdouble cZFermionAntitriplet_3(double logY, const thermo &thArgs);
cdouble cZBosonTriplet_1(double logY, const thermo &thArgs);
cdouble cZBosonTriplet_2(double logY, const thermo &thArgs);
cdouble cZBosonTriplet_3(double logY, const thermo &thArgs);
cdouble cZBosonAntitriplet_1(double logY, const thermo &thArgs);
cdouble cZBosonAntitriplet_2(double logY, const thermo &thArgs);
cdouble cZBosonAntitriplet_3(double logY, const thermo &thArgs);
cdouble cFFermionTriplet_1(double logY, const thermo &thArgs);
cdouble cFFermionTriplet_2(double logY, const thermo &thArgs);
cdouble cFFermionAntitriplet_1(double logY, const thermo &thArgs);
cdouble cFFermionAntitriplet_2(double logY, const thermo &thArgs);
cdouble cFBosonTriplet_1(double logY, const thermo &thArgs);
cdouble cFBosonTriplet_2(double logY, const thermo &thArgs);
cdouble cFBosonAntitriplet_1(double logY, const thermo &thArgs);
cdouble cFBosonAntitriplet_2(double logY, const thermo &thArgs);
cdouble cdFdMBosonTriplet_aux(double p, const particle &pArgs, const thermo &thArgs);
cdouble cdFdMBosonAntitriplet_aux(double p, const particle &pArgs, const thermo &thArgs);
cdouble cdFdMFermionTriplet_aux(double p, const particle &pArgs, const thermo &thArgs);
cdouble cdFdMFermionAntitriplet_aux(double p, const particle &pArgs, const thermo &thArgs);

//External methods
double cEn(double p, const particle &pArgs);
double cLogY(double p, const particle &pArgs, const thermo &thArgs);

cdouble cZFermionTriplet(double p, const particle &pArgs, const thermo &thArgs);
cdouble cZFermionAntitriplet(double p, const particle &pArgs, const thermo &thArgs);
cdouble cZBosonTriplet(double p, const particle &pArgs, const thermo &thArgs);
cdouble cZBosonAntitriplet(double p, const particle &pArgs, const thermo &thArgs);

double cFFermionSinglet(double p, const particle &pArgs, const thermo &thArgs);
double cFBosonSinglet(double p, const particle &pArgs, const thermo &thArgs);
cdouble cFFermionTriplet(double p, const particle &pArgs, const thermo &thArgs);
cdouble cFFermionAntitriplet(double p, const particle &pArgs, const thermo &thArgs);
cdouble cFBosonTriplet(double p, const particle &pArgs, const thermo &thArgs);
cdouble cFBosonAntitriplet(double p, const particle &pArgs, const thermo &thArgs);

double cdFdMBosonSinglet(double p, const particle &pArgs, const thermo &thArgs);
double cdFdMFermionSinglet(double p, const particle &pArgs, const thermo &thArgs);
cdouble cdFdMBosonTriplet(double p, const particle &pArgs, const thermo &thArgs);
cdouble cdFdMBosonAntitriplet(double p, const particle &pArgs, const thermo &thArgs);
cdouble cdFdMFermionTriplet(double p, const particle &pArgs, const thermo &thArgs);
cdouble cdFdMFermionAntitriplet(double p, const particle &pArgs, const thermo &thArgs);

#endif