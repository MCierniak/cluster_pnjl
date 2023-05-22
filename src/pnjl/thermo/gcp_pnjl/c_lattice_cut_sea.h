#ifndef LATTICE_CUT_SEA_H
#define LATTICE_CUT_SEA_H

#include "../c_distributions.h"

//Internal methods
double cTMott(const thermo &thArgs);
double cTC(const thermo &thArgs);
double cDeltaLS(const thermo &thArgs);

double cPressureQLIntegrandReal(double p, void *pars);
double cPressureQSIntegrandReal(double p, void *pars);

double cBDensityQLIntegrandReal(double p, void *pars);
double cBDensityQSIntegrandReal(double p, void *pars);

double cSDensityQLIntegrandReal(double p, void *pars);
double cSDensityQSIntegrandReal(double p, void *pars);

//External methods
double cML(const thermo &thArgs);
double cMS(const thermo &thArgs);

double cPressureSea(const thermo &thArgs);
double cBDensitySea(const thermo &thArgs);
double cQNumberCumulantSea(const thermo &thArgs);
double cSDensitySea(const thermo &thArgs);

double cPressureField(const thermo &thArgs);
double cBDensityField(const thermo &thArgs);
double cQNumberCumulantField(const thermo &thArgs);
double cSDensityField(const thermo &thArgs);

double cPressureQ(char typ, thermo thArgs);
double cBDensityQ(char typ, thermo thArgs);
double cQNumberCumulantQ(char typ, thermo thArgs);
double cSDensityQ(char typ, double p, double nB, thermo thArgs);

#endif