"""### Description
(needs updating)

### Functions
(needs updating)
"""


import math

import scipy.optimize
import scipy.integrate


EXP_LIMIT = 709.78271


MI = {
    'pi0': 135.0, 'pi': 140.0, 'K': 494.0, 'K0': 498.0, 'eta': 548.0,
    'rho': 775.0, 'omega': 783.0, 'K*(892)': 892.0, 'K*0(892)': 896.0, 'p': 938.0,
    'n': 940.0, 'etaPrime': 958.0, 'a0': 980.0, 'f0': 990.0, 'phi': 1019.0,
    'Lambda': 1116.0, 'h1': 1170.0, 'Sigma+': 1189.0, 'Sigma0': 1193.0,
    'Sigma-': 1197.0, 'b1': 1230.0, 'a1': 1230.0, 'Delta': 1232.0,
    'K1(1270)': 1272.0, 'f2': 1275.0, 'f1': 1282.0, 'eta(1295)': 1294.0,
    'pi(1300)': 1300.0, 'Xi0': 1315.0, 'a2': 1318.0, 'Xi-': 1322.0,
    'f0(1370)': 1350.0, 'pi1(1400)': 1354.0, 'Sigma(1385)': 1385.0,
    'K1(1400)': 1403.0, 'Lambda(1405)': 1405.0, 'eta(1405)': 1409.0,
    'K*(1410)': 1414.0, 'omega(1420)': 1425.0, 'K0*(1430)': 1425.0,
    'K2*(1430)': 1426.0, 'f1(1420)': 1426.0, 'K2*0(1430)': 1432.0,
    'N(1440)': 1440.0, 'rho(1450)': 1465.0, 'a0(1450)': 1474.0,
    'eta(1475)': 1476.0, 'f0(1500)': 1505.0, 'Lambda(1520)': 1520.0,
    'N(1520)': 1520.0, 'f2Prime(1525)': 1525.0, 'Xi0(1530)': 1532.0,
    'N(1535)': 1535.0, 'Xi-(1530)': 1535.0, 'Delta(1600)': 1600.0,
    'Lambda(1600)': 1600.0, 'eta2(1645)': 1617.0, 'Delta(1620)': 1630.0,
    'N(1650)': 1655.0, 'Sigma(1660)': 1660.0, 'pi1(1600)': 1662.0,
    'omega3(1670)': 1667.0, 'omega(1650)': 1670.0, 'Lambda(1670)': 1670.0,
    'Sigma(1670)': 1670.0, 'pi2(1670)': 1672.0, 'Omega-': 1673.0,
    'N(1675)': 1675.0, 'phi(1680)': 1680.0, 'N(1680)': 1685.0,
    'rho3(1690)': 1689.0, 'Lambda(1690)': 1690.0, 'Xi(1690)': 1690.0,
    'N(1700)': 1700.0, 'Delta(1700)': 1700.0, 'N(1710)': 1710.0,
    'K*(1680)': 1717.0, 'rho(1700)': 1720.0, 'f0(1710)': 1720.0,
    'N(1720)': 1720.0, 'Sigma(1750)': 1750.0, 'K2(1770)': 1773.0,
    'Sigma(1775)': 1775.0, 'K3*(1780)': 1776.0, 'Lambda(1800)': 1800.0,
    'Lambda(1810)': 1810.0, 'pi(1800)': 1812.0, 'K2(1820)': 1816.0,
    'Lambda(1820)': 1820.0, 'Xi(1820)': 1823.0, 'Lambda(1830)': 1830.0,
    'phi3(1850)': 1854.0, 'N(1875)': 1875.0, 'Delta(1905)': 1880.0,
    'Delta(1910)': 1890.0, 'Lambda(1890)': 1890.0, 'pi2(1880)': 1895.0,
    'N(1900)': 1900.0, 'Sigma(1915)': 1915.0, 'Delta(1920)': 1920.0,
    'Delta(1950)': 1930.0, 'Sigma(1940)': 1940.0, 'f2(1950)': 1944.0,
    'Delta(1930)': 1950.0, 'Xi(1950)': 1950.0, 'a4(2040)': 1996.0,
    'f2(2010)': 2011.0, 'f4(2050)': 2018.0, 'Xi(2030)': 2025.0,
    'Sigma(2030)': 2030.0, 'K4*(2045)': 2045.0, 'Lambda(2100)': 2100.0,
    'Lambda(2110)': 2110.0, 'phi(2170)': 2175.0, 'N(2190)': 2190.0,
    'N(2200)': 2250.0, 'Sigma(2250)': 2250.0, 'Omega-(2250)': 2252.0,
    'N(2250)': 2275.0, 'f2(2300)': 2297.0, 'f2(2340)': 2339.0,
    'Lambda(2350)': 2350.0, 'Delta(2420)': 2420.0, 'N(2600)': 2600.0
}

DI = {
    'pi0': 1.0, 'pi': 2.0, 'K': 2.0, 'K0': 2.0, 'eta': 1.0,
    'rho': 9.0, 'omega': 3.0, 'K*(892)': 6.0, 'K*0(892)': 6.0, 'p': 2.0,
    'n': 2.0, 'etaPrime': 1.0, 'a0': 3.0, 'f0': 1.0, 'phi': 3.0,
    'Lambda': 2.0, 'h1': 3.0, 'Sigma+': 2.0, 'Sigma0': 2.0,
    'Sigma-': 2.0, 'b1': 9.0, 'a1': 9.0, 'Delta': 16.0,
    'K1(1270)': 12.0, 'f2': 5.0, 'f1': 3.0, 'eta(1295)': 1.0,
    'pi(1300)': 3.0, 'Xi0': 2.0, 'a2': 15.0, 'Xi-': 2.0,
    'f0(1370)': 1.0, 'pi1(1400)': 9.0, 'Sigma(1385)': 12.0,
    'K1(1400)': 12.0, 'Lambda(1405)': 2.0, 'eta(1405)': 1.0,
    'K*(1410)': 12.0, 'omega(1420)': 3.0, 'K0*(1430)': 4.0,
    'K2*(1430)': 10.0, 'f1(1420)': 3.0, 'K2*0(1430)': 10.0,
    'N(1440)': 4.0, 'rho(1450)': 9.0, 'a0(1450)': 3.0,
    'eta(1475)': 1.0, 'f0(1500)': 1.0, 'Lambda(1520)': 4.0,
    'N(1520)': 8.0, 'f2Prime(1525)': 5.0, 'Xi0(1530)': 4.0,
    'N(1535)': 4.0, 'Xi-(1530)': 4.0, 'Delta(1600)': 16.0,
    'Lambda(1600)': 2.0, 'eta2(1645)': 5.0, 'Delta(1620)': 8.0,
    'N(1650)': 4.0, 'Sigma(1660)': 6.0, 'pi1(1600)': 9.0,
    'omega3(1670)': 7.0, 'omega(1650)': 3.0, 'Lambda(1670)': 2.0,
    'Sigma(1670)': 12.0, 'pi2(1670)': 15.0, 'Omega-': 4.0,
    'N(1675)': 12.0, 'phi(1680)': 3.0, 'N(1680)': 12.0,
    'rho3(1690)': 21.0, 'Lambda(1690)': 4.0, 'Xi(1690)': 4.0,
    'N(1700)': 8.0, 'Delta(1700)': 16.0, 'N(1710)': 4.0,
    'K*(1680)': 12.0, 'rho(1700)': 9.0, 'f0(1710)': 1.0,
    'N(1720)': 8.0, 'Sigma(1750)': 6.0, 'K2(1770)': 20.0,
    'Sigma(1775)': 18.0, 'K3*(1780)': 28.0, 'Lambda(1800)': 2.0,
    'Lambda(1810)': 2.0, 'pi(1800)': 3.0, 'K2(1820)': 20.0,
    'Lambda(1820)': 6.0, 'Xi(1820)': 8.0, 'Lambda(1830)': 6.0,
    'phi3(1850)': 7.0, 'N(1875)': 8.0, 'Delta(1905)': 24.0,
    'Delta(1910)': 8.0, 'Lambda(1890)': 4.0, 'pi2(1880)': 15.0,
    'N(1900)': 8.0, 'Sigma(1915)': 18.0, 'Delta(1920)': 16.0,
    'Delta(1950)': 32.0, 'Sigma(1940)': 12.0, 'f2(1950)': 5.0,
    'Delta(1930)': 24.0, 'Xi(1950)': 4.0, 'a4(2040)': 27.0,
    'f2(2010)': 5.0, 'f4(2050)': 9.0, 'Xi(2030)': 12.0,
    'Sigma(2030)': 24.0, 'K4*(2045)': 36.0, 'Lambda(2100)': 8.0,
    'Lambda(2110)': 6.0, 'phi(2170)': 3.0, 'N(2190)': 16.0,
    'N(2200)': 20.0, 'Sigma(2250)': 6.0, 'Omega-(2250)': 2.0,
    'N(2250)': 20.0, 'f2(2300)': 5.0, 'f2(2340)': 5.0,
    'Lambda(2350)': 10.0, 'Delta(2420)': 48.0, 'N(2600)': 24.0,
}
    
BI = {
    'pi0': 0.0, 'pi': 0.0, 'K': 0.0, 'K0': 0.0, 'eta': 0.0,
    'rho': 0.0, 'omega': 0.0, 'K*(892)': 0.0, 'K*0(892)': 0.0, 'p': 1.0,
    'n': 1.0, 'etaPrime': 0.0, 'a0': 0.0, 'f0': 0.0, 'phi': 0.0,
    'Lambda': 1.0, 'h1': 0.0, 'Sigma+': 1.0, 'Sigma0': 1.0,
    'Sigma-': 1.0, 'b1': 0.0, 'a1': 0.0, 'Delta': 1.0,
    'K1(1270)': 0.0, 'f2': 0.0, 'f1': 0.0, 'eta(1295)': 0.0,
    'pi(1300)': 0.0, 'Xi0': 1.0, 'a2': 0.0, 'Xi-': 1.0,
    'f0(1370)': 0.0, 'pi1(1400)': 0.0, 'Sigma(1385)': 1.0,
    'K1(1400)': 0.0, 'Lambda(1405)': 1.0, 'eta(1405)': 0.0,
    'K*(1410)': 0.0, 'omega(1420)': 0.0, 'K0*(1430)': 0.0,
    'K2*(1430)': 0.0, 'f1(1420)': 0.0, 'K2*0(1430)': 0.0,
    'N(1440)': 1.0, 'rho(1450)': 0.0, 'a0(1450)': 0.0,
    'eta(1475)': 0.0, 'f0(1500)': 0.0, 'Lambda(1520)': 1.0,
    'N(1520)': 1.0, 'f2Prime(1525)': 0.0, 'Xi0(1530)': 1.0,
    'N(1535)': 1.0, 'Xi-(1530)': 1.0, 'Delta(1600)': 1.0,
    'Lambda(1600)': 1.0, 'eta2(1645)': 0.0, 'Delta(1620)': 1.0,
    'N(1650)': 1.0, 'Sigma(1660)': 1.0, 'pi1(1600)': 0.0,
    'omega3(1670)': 0.0, 'omega(1650)': 0.0, 'Lambda(1670)': 1.0,
    'Sigma(1670)': 1.0, 'pi2(1670)': 0.0, 'Omega-': 1.0,
    'N(1675)': 1.0, 'phi(1680)': 0.0, 'N(1680)': 1.0,
    'rho3(1690)': 0.0, 'Lambda(1690)': 1.0, 'Xi(1690)': 1.0,
    'N(1700)': 1.0, 'Delta(1700)': 1.0, 'N(1710)': 1.0,
    'K*(1680)': 0.0, 'rho(1700)': 0.0, 'f0(1710)': 0.0,
    'N(1720)': 1.0, 'Sigma(1750)': 1.0, 'K2(1770)': 0.0,
    'Sigma(1775)': 1.0, 'K3*(1780)': 0.0, 'Lambda(1800)': 1.0,
    'Lambda(1810)': 1.0, 'pi(1800)': 0.0, 'K2(1820)': 0.0,
    'Lambda(1820)': 1.0, 'Xi(1820)': 1.0, 'Lambda(1830)': 1.0,
    'phi3(1850)': 0.0, 'N(1875)': 1.0, 'Delta(1905)': 1.0,
    'Delta(1910)': 1.0, 'Lambda(1890)': 1.0, 'pi2(1880)': 0.0,
    'N(1900)': 1.0, 'Sigma(1915)': 1.0, 'Delta(1920)': 1.0,
    'Delta(1950)': 1.0, 'Sigma(1940)': 1.0, 'f2(1950)': 0.0,
    'Delta(1930)': 1.0, 'Xi(1950)': 1.0, 'a4(2040)': 0.0,
    'f2(2010)': 0.0, 'f4(2050)': 0.0, 'Xi(2030)': 1.0,
    'Sigma(2030)': 1.0, 'K4*(2045)': 0.0, 'Lambda(2100)': 1.0,
    'Lambda(2110)': 1.0, 'phi(2170)': 0.0, 'N(2190)': 1.0,
    'N(2200)': 1.0, 'Sigma(2250)': 1.0, 'Omega-(2250)': 1.0,
    'N(2250)': 1.0, 'f2(2300)': 0.0, 'f2(2340)': 0.0,
    'Lambda(2350)': 1.0, 'Delta(2420)': 1.0, 'N(2600)': 1.0,
}


def En(p: float, mass: float) -> float:
    body = math.fsum([p**2, mass**2])
    return math.sqrt(body)


def log_y_p(p: float, T: float, muB: float, mass: float, B: float) -> float:
    ensum = math.fsum([En(p, mass), -B*muB])
    return ensum/T


def log_y_m(p: float, T: float, muB: float, mass: float, B: float) -> float:
    ensum = math.fsum([En(p, mass), B*muB])
    return ensum/T


def f_fermion_p(p: float, T: float, muB: float, mass: float, B: float) -> float:
    logy = log_y_p(p, T, muB, mass, B)
    if logy >= EXP_LIMIT:
        return 0.0
    else:
        return 1.0/math.fsum([math.exp(logy), 1.0])
    

def f_fermion_m(p: float, T: float, muB: float, mass: float, B: float) -> float:
    logy = log_y_m(p, T, muB, mass, B)
    if logy >= EXP_LIMIT:
        return 0.0
    else:
        return 1.0/math.fsum([math.exp(logy), 1.0])


def f_boson_p(p: float, T: float, muB: float, mass: float, B: float) -> float:
    logy = log_y_p(p, T, muB, mass, B)
    if logy >= EXP_LIMIT:
        return 0.0
    else:
        return 1.0/math.expm1(logy)
    

def f_boson_m(p: float, T: float, muB: float, mass: float, B: float) -> float:
    logy = log_y_m(p, T, muB, mass, B)
    if logy >= EXP_LIMIT:
        return 0.0
    else:
        return 1.0/math.expm1(logy)
    

def fermion_pressure_integ(p: float, T: float, muB: float, M: float, B: float) -> float:
    return -((p**4)/En(p, M))*(f_fermion_p(p, T, muB, M, B) + f_fermion_m(p, T, muB, M, B))


def boson_pressure_integ(p: float, T: float, muB: float, M: float, B: float) -> float:
    return 0.5*((p**4)/En(p, M))*(f_boson_p(p, T, muB, M, B) + f_boson_m(p, T, muB, M, B))
    

def boson_sdensity_integ(p: float, T: float, muB: float, M: float, B: float) -> float:
    fp = f_boson_p(p, T, muB, M, B)
    fn = f_boson_m(p, T, muB, M, B)
    positive = 0.0
    negative = 0.0
    if fp != 0.0:
        positive = -fp*math.log(fp) + (1.0 + fp)*math.log1p(fp)
    if fn != 0.0:
        negative = -fn*math.log(fn) + (1.0 + fn)*math.log1p(fn)
    return 0.5*(p**2)*(positive + negative)


def fermion_sdensity_integ(p: float, T: float, muB: float, M: float, B: float) -> float:
    fp = f_fermion_p(p, T, muB, M, B)
    fn = f_fermion_m(p, T, muB, M, B)
    positive = 0.0
    negative = 0.0
    if fp != 0.0:
        positive = -fp*math.log(fp) - (1.0 - fp)*math.log(1.0 - fp)
    if fn != 0.0:
        negative = -fn*math.log(fn) - (1.0 - fn)*math.log(1.0 - fn)
    return (p**2)*(positive + negative)


def boson_bdensity_integ(p: float, T: float, muB: float, M: float, B: float) -> float:
    fp = f_boson_p(p, T, muB, M, B)
    fn = f_boson_m(p, T, muB, M, B)
    return -0.5*(p**2)*(fp - fn)


def fermion_bdensity_integ(p: float, T: float, muB: float, M: float, B: float) -> float:
    fp = f_fermion_p(p, T, muB, M, B)
    fn = f_fermion_m(p, T, muB, M, B)
    return (p**2)*(fp + fn)


pressure_hash = {
    'pi0': boson_pressure_integ, 'pi': boson_pressure_integ,
    'K': boson_pressure_integ, 'K0': boson_pressure_integ,
    'eta': boson_pressure_integ, 'rho': boson_pressure_integ,
    'omega': boson_pressure_integ, 'K*(892)': boson_pressure_integ,
    'K*0(892)': boson_pressure_integ, 'p': fermion_pressure_integ,
    'n': fermion_pressure_integ, 'etaPrime': boson_pressure_integ,
    'a0': boson_pressure_integ, 'f0': boson_pressure_integ,
    'phi': boson_pressure_integ, 'Lambda': fermion_pressure_integ,
    'h1': boson_pressure_integ, 'Sigma+': fermion_pressure_integ,
    'Sigma0': fermion_pressure_integ, 'Sigma-': fermion_pressure_integ,
    'b1': boson_pressure_integ, 'a1': boson_pressure_integ,
    'Delta': fermion_pressure_integ, 'K1(1270)': boson_pressure_integ,
    'f2': boson_pressure_integ, 'f1': boson_pressure_integ,
    'eta(1295)': boson_pressure_integ, 'pi(1300)': boson_pressure_integ,
    'Xi0': fermion_pressure_integ, 'a2': boson_pressure_integ,
    'Xi-': fermion_pressure_integ, 'f0(1370)': boson_pressure_integ,
    'pi1(1400)': boson_pressure_integ, 'Sigma(1385)': fermion_pressure_integ,
    'K1(1400)': boson_pressure_integ, 'Lambda(1405)': fermion_pressure_integ,
    'eta(1405)': boson_pressure_integ, 'K*(1410)': boson_pressure_integ,
    'omega(1420)': boson_pressure_integ, 'K0*(1430)': boson_pressure_integ,
    'K2*(1430)': boson_pressure_integ, 'f1(1420)': boson_pressure_integ,
    'K2*0(1430)': boson_pressure_integ, 'N(1440)': fermion_pressure_integ,
    'rho(1450)': boson_pressure_integ, 'a0(1450)': boson_pressure_integ,
    'eta(1475)': boson_pressure_integ, 'f0(1500)': boson_pressure_integ,
    'Lambda(1520)': fermion_pressure_integ, 'N(1520)': fermion_pressure_integ,
    'f2Prime(1525)': boson_pressure_integ, 'Xi0(1530)': fermion_pressure_integ,
    'N(1535)': fermion_pressure_integ, 'Xi-(1530)': fermion_pressure_integ,
    'Delta(1600)': fermion_pressure_integ, 'Lambda(1600)': fermion_pressure_integ,
    'eta2(1645)': boson_pressure_integ, 'Delta(1620)': fermion_pressure_integ,
    'N(1650)': fermion_pressure_integ, 'Sigma(1660)': fermion_pressure_integ,
    'pi1(1600)': boson_pressure_integ, 'omega3(1670)': boson_pressure_integ,
    'omega(1650)': boson_pressure_integ, 'Lambda(1670)': fermion_pressure_integ,
    'Sigma(1670)': fermion_pressure_integ, 'pi2(1670)': boson_pressure_integ,
    'Omega-': fermion_pressure_integ, 'N(1675)': fermion_pressure_integ,
    'phi(1680)': boson_pressure_integ, 'N(1680)': fermion_pressure_integ,
    'rho3(1690)': boson_pressure_integ, 'Lambda(1690)': fermion_pressure_integ,
    'Xi(1690)': fermion_pressure_integ, 'N(1700)': fermion_pressure_integ,
    'Delta(1700)': fermion_pressure_integ, 'N(1710)': fermion_pressure_integ,
    'K*(1680)': boson_pressure_integ, 'rho(1700)': boson_pressure_integ,
    'f0(1710)': boson_pressure_integ, 'N(1720)': fermion_pressure_integ,
    'Sigma(1750)': fermion_pressure_integ, 'K2(1770)': boson_pressure_integ,
    'Sigma(1775)': fermion_pressure_integ, 'K3*(1780)': boson_pressure_integ,
    'Lambda(1800)': fermion_pressure_integ, 'Lambda(1810)': fermion_pressure_integ,
    'pi(1800)': boson_pressure_integ, 'K2(1820)': boson_pressure_integ,
    'Lambda(1820)': fermion_pressure_integ, 'Xi(1820)': fermion_pressure_integ,
    'Lambda(1830)': fermion_pressure_integ, 'phi3(1850)': boson_pressure_integ,
    'N(1875)': fermion_pressure_integ, 'Delta(1905)': fermion_pressure_integ,
    'Delta(1910)': fermion_pressure_integ, 'Lambda(1890)': fermion_pressure_integ,
    'pi2(1880)': boson_pressure_integ, 'N(1900)': fermion_pressure_integ,
    'Sigma(1915)': fermion_pressure_integ, 'Delta(1920)': fermion_pressure_integ,
    'Delta(1950)': fermion_pressure_integ, 'Sigma(1940)': fermion_pressure_integ,
    'f2(1950)': boson_pressure_integ, 'Delta(1930)': fermion_pressure_integ,
    'Xi(1950)': fermion_pressure_integ, 'a4(2040)': boson_pressure_integ,
    'f2(2010)': boson_pressure_integ, 'f4(2050)': boson_pressure_integ,
    'Xi(2030)': fermion_pressure_integ, 'Sigma(2030)': fermion_pressure_integ,
    'K4*(2045)': boson_pressure_integ, 'Lambda(2100)': fermion_pressure_integ,
    'Lambda(2110)': fermion_pressure_integ, 'phi(2170)': boson_pressure_integ,
    'N(2190)': fermion_pressure_integ, 'N(2200)': fermion_pressure_integ,
    'Sigma(2250)': fermion_pressure_integ, 'Omega-(2250)': fermion_pressure_integ,
    'N(2250)': fermion_pressure_integ, 'f2(2300)': boson_pressure_integ,
    'f2(2340)': boson_pressure_integ, 'Lambda(2350)': fermion_pressure_integ,
    'Delta(2420)': fermion_pressure_integ, 'N(2600)': fermion_pressure_integ,
}

sdensity_hash = {
    'pi0': boson_sdensity_integ, 'pi': boson_sdensity_integ,
    'K': boson_sdensity_integ, 'K0': boson_sdensity_integ,
    'eta': boson_sdensity_integ, 'rho': boson_sdensity_integ,
    'omega': boson_sdensity_integ, 'K*(892)': boson_sdensity_integ,
    'K*0(892)': boson_sdensity_integ, 'p': fermion_sdensity_integ,
    'n': fermion_sdensity_integ, 'etaPrime': boson_sdensity_integ,
    'a0': boson_sdensity_integ, 'f0': boson_sdensity_integ,
    'phi': boson_sdensity_integ, 'Lambda': fermion_sdensity_integ,
    'h1': boson_sdensity_integ, 'Sigma+': fermion_sdensity_integ,
    'Sigma0': fermion_sdensity_integ, 'Sigma-': fermion_sdensity_integ,
    'b1': boson_sdensity_integ, 'a1': boson_sdensity_integ,
    'Delta': fermion_sdensity_integ, 'K1(1270)': boson_sdensity_integ,
    'f2': boson_sdensity_integ, 'f1': boson_sdensity_integ,
    'eta(1295)': boson_sdensity_integ, 'pi(1300)': boson_sdensity_integ,
    'Xi0': fermion_sdensity_integ, 'a2': boson_sdensity_integ,
    'Xi-': fermion_sdensity_integ, 'f0(1370)': boson_sdensity_integ,
    'pi1(1400)': boson_sdensity_integ, 'Sigma(1385)': fermion_sdensity_integ,
    'K1(1400)': boson_sdensity_integ, 'Lambda(1405)': fermion_sdensity_integ,
    'eta(1405)': boson_sdensity_integ, 'K*(1410)': boson_sdensity_integ,
    'omega(1420)': boson_sdensity_integ, 'K0*(1430)': boson_sdensity_integ,
    'K2*(1430)': boson_sdensity_integ, 'f1(1420)': boson_sdensity_integ,
    'K2*0(1430)': boson_sdensity_integ, 'N(1440)': fermion_sdensity_integ,
    'rho(1450)': boson_sdensity_integ, 'a0(1450)': boson_sdensity_integ,
    'eta(1475)': boson_sdensity_integ, 'f0(1500)': boson_sdensity_integ,
    'Lambda(1520)': fermion_sdensity_integ, 'N(1520)': fermion_sdensity_integ,
    'f2Prime(1525)': boson_sdensity_integ, 'Xi0(1530)': fermion_sdensity_integ,
    'N(1535)': fermion_sdensity_integ, 'Xi-(1530)': fermion_sdensity_integ,
    'Delta(1600)': fermion_sdensity_integ, 'Lambda(1600)': fermion_sdensity_integ,
    'eta2(1645)': boson_sdensity_integ, 'Delta(1620)': fermion_sdensity_integ,
    'N(1650)': fermion_sdensity_integ, 'Sigma(1660)': fermion_sdensity_integ,
    'pi1(1600)': boson_sdensity_integ, 'omega3(1670)': boson_sdensity_integ,
    'omega(1650)': boson_sdensity_integ, 'Lambda(1670)': fermion_sdensity_integ,
    'Sigma(1670)': fermion_sdensity_integ, 'pi2(1670)': boson_sdensity_integ,
    'Omega-': fermion_sdensity_integ, 'N(1675)': fermion_sdensity_integ,
    'phi(1680)': boson_sdensity_integ, 'N(1680)': fermion_sdensity_integ,
    'rho3(1690)': boson_sdensity_integ, 'Lambda(1690)': fermion_sdensity_integ,
    'Xi(1690)': fermion_sdensity_integ, 'N(1700)': fermion_sdensity_integ,
    'Delta(1700)': fermion_sdensity_integ, 'N(1710)': fermion_sdensity_integ,
    'K*(1680)': boson_sdensity_integ, 'rho(1700)': boson_sdensity_integ,
    'f0(1710)': boson_sdensity_integ, 'N(1720)': fermion_sdensity_integ,
    'Sigma(1750)': fermion_sdensity_integ, 'K2(1770)': boson_sdensity_integ,
    'Sigma(1775)': fermion_sdensity_integ, 'K3*(1780)': boson_sdensity_integ,
    'Lambda(1800)': fermion_sdensity_integ, 'Lambda(1810)': fermion_sdensity_integ,
    'pi(1800)': boson_sdensity_integ, 'K2(1820)': boson_sdensity_integ,
    'Lambda(1820)': fermion_sdensity_integ, 'Xi(1820)': fermion_sdensity_integ,
    'Lambda(1830)': fermion_sdensity_integ, 'phi3(1850)': boson_sdensity_integ,
    'N(1875)': fermion_sdensity_integ, 'Delta(1905)': fermion_sdensity_integ,
    'Delta(1910)': fermion_sdensity_integ, 'Lambda(1890)': fermion_sdensity_integ,
    'pi2(1880)': boson_sdensity_integ, 'N(1900)': fermion_sdensity_integ,
    'Sigma(1915)': fermion_sdensity_integ, 'Delta(1920)': fermion_sdensity_integ,
    'Delta(1950)': fermion_sdensity_integ, 'Sigma(1940)': fermion_sdensity_integ,
    'f2(1950)': boson_sdensity_integ, 'Delta(1930)': fermion_sdensity_integ,
    'Xi(1950)': fermion_sdensity_integ, 'a4(2040)': boson_sdensity_integ,
    'f2(2010)': boson_sdensity_integ, 'f4(2050)': boson_sdensity_integ,
    'Xi(2030)': fermion_sdensity_integ, 'Sigma(2030)': fermion_sdensity_integ,
    'K4*(2045)': boson_sdensity_integ, 'Lambda(2100)': fermion_sdensity_integ,
    'Lambda(2110)': fermion_sdensity_integ, 'phi(2170)': boson_sdensity_integ,
    'N(2190)': fermion_sdensity_integ, 'N(2200)': fermion_sdensity_integ,
    'Sigma(2250)': fermion_sdensity_integ, 'Omega-(2250)': fermion_sdensity_integ,
    'N(2250)': fermion_sdensity_integ, 'f2(2300)': boson_sdensity_integ,
    'f2(2340)': boson_sdensity_integ, 'Lambda(2350)': fermion_sdensity_integ,
    'Delta(2420)': fermion_sdensity_integ, 'N(2600)': fermion_sdensity_integ,
}

bdensity_hash = {
    'pi0': boson_bdensity_integ, 'pi': boson_bdensity_integ,
    'K': boson_bdensity_integ, 'K0': boson_bdensity_integ,
    'eta': boson_bdensity_integ, 'rho': boson_bdensity_integ,
    'omega': boson_bdensity_integ, 'K*(892)': boson_bdensity_integ,
    'K*0(892)': boson_bdensity_integ, 'p': fermion_bdensity_integ,
    'n': fermion_bdensity_integ, 'etaPrime': boson_bdensity_integ,
    'a0': boson_bdensity_integ, 'f0': boson_bdensity_integ,
    'phi': boson_bdensity_integ, 'Lambda': fermion_bdensity_integ,
    'h1': boson_bdensity_integ, 'Sigma+': fermion_bdensity_integ,
    'Sigma0': fermion_bdensity_integ, 'Sigma-': fermion_bdensity_integ,
    'b1': boson_bdensity_integ, 'a1': boson_bdensity_integ,
    'Delta': fermion_bdensity_integ, 'K1(1270)': boson_bdensity_integ,
    'f2': boson_bdensity_integ, 'f1': boson_bdensity_integ,
    'eta(1295)': boson_bdensity_integ, 'pi(1300)': boson_bdensity_integ,
    'Xi0': fermion_bdensity_integ, 'a2': boson_bdensity_integ,
    'Xi-': fermion_bdensity_integ, 'f0(1370)': boson_bdensity_integ,
    'pi1(1400)': boson_bdensity_integ, 'Sigma(1385)': fermion_bdensity_integ,
    'K1(1400)': boson_bdensity_integ, 'Lambda(1405)': fermion_bdensity_integ,
    'eta(1405)': boson_bdensity_integ, 'K*(1410)': boson_bdensity_integ,
    'omega(1420)': boson_bdensity_integ, 'K0*(1430)': boson_bdensity_integ,
    'K2*(1430)': boson_bdensity_integ, 'f1(1420)': boson_bdensity_integ,
    'K2*0(1430)': boson_bdensity_integ, 'N(1440)': fermion_bdensity_integ,
    'rho(1450)': boson_bdensity_integ, 'a0(1450)': boson_bdensity_integ,
    'eta(1475)': boson_bdensity_integ, 'f0(1500)': boson_bdensity_integ,
    'Lambda(1520)': fermion_bdensity_integ, 'N(1520)': fermion_bdensity_integ,
    'f2Prime(1525)': boson_bdensity_integ, 'Xi0(1530)': fermion_bdensity_integ,
    'N(1535)': fermion_bdensity_integ, 'Xi-(1530)': fermion_bdensity_integ,
    'Delta(1600)': fermion_bdensity_integ, 'Lambda(1600)': fermion_bdensity_integ,
    'eta2(1645)': boson_bdensity_integ, 'Delta(1620)': fermion_bdensity_integ,
    'N(1650)': fermion_bdensity_integ, 'Sigma(1660)': fermion_bdensity_integ,
    'pi1(1600)': boson_bdensity_integ, 'omega3(1670)': boson_bdensity_integ,
    'omega(1650)': boson_bdensity_integ, 'Lambda(1670)': fermion_bdensity_integ,
    'Sigma(1670)': fermion_bdensity_integ, 'pi2(1670)': boson_bdensity_integ,
    'Omega-': fermion_bdensity_integ, 'N(1675)': fermion_bdensity_integ,
    'phi(1680)': boson_bdensity_integ, 'N(1680)': fermion_bdensity_integ,
    'rho3(1690)': boson_bdensity_integ, 'Lambda(1690)': fermion_bdensity_integ,
    'Xi(1690)': fermion_bdensity_integ, 'N(1700)': fermion_bdensity_integ,
    'Delta(1700)': fermion_bdensity_integ, 'N(1710)': fermion_bdensity_integ,
    'K*(1680)': boson_bdensity_integ, 'rho(1700)': boson_bdensity_integ,
    'f0(1710)': boson_bdensity_integ, 'N(1720)': fermion_bdensity_integ,
    'Sigma(1750)': fermion_bdensity_integ, 'K2(1770)': boson_bdensity_integ,
    'Sigma(1775)': fermion_bdensity_integ, 'K3*(1780)': boson_bdensity_integ,
    'Lambda(1800)': fermion_bdensity_integ, 'Lambda(1810)': fermion_bdensity_integ,
    'pi(1800)': boson_bdensity_integ, 'K2(1820)': boson_bdensity_integ,
    'Lambda(1820)': fermion_bdensity_integ, 'Xi(1820)': fermion_bdensity_integ,
    'Lambda(1830)': fermion_bdensity_integ, 'phi3(1850)': boson_bdensity_integ,
    'N(1875)': fermion_bdensity_integ, 'Delta(1905)': fermion_bdensity_integ,
    'Delta(1910)': fermion_bdensity_integ, 'Lambda(1890)': fermion_bdensity_integ,
    'pi2(1880)': boson_bdensity_integ, 'N(1900)': fermion_bdensity_integ,
    'Sigma(1915)': fermion_bdensity_integ, 'Delta(1920)': fermion_bdensity_integ,
    'Delta(1950)': fermion_bdensity_integ, 'Sigma(1940)': fermion_bdensity_integ,
    'f2(1950)': boson_bdensity_integ, 'Delta(1930)': fermion_bdensity_integ,
    'Xi(1950)': fermion_bdensity_integ, 'a4(2040)': boson_bdensity_integ,
    'f2(2010)': boson_bdensity_integ, 'f4(2050)': boson_bdensity_integ,
    'Xi(2030)': fermion_bdensity_integ, 'Sigma(2030)': fermion_bdensity_integ,
    'K4*(2045)': boson_bdensity_integ, 'Lambda(2100)': fermion_bdensity_integ,
    'Lambda(2110)': fermion_bdensity_integ, 'phi(2170)': boson_bdensity_integ,
    'N(2190)': fermion_bdensity_integ, 'N(2200)': fermion_bdensity_integ,
    'Sigma(2250)': fermion_bdensity_integ, 'Omega-(2250)': fermion_bdensity_integ,
    'N(2250)': fermion_bdensity_integ, 'f2(2300)': boson_bdensity_integ,
    'f2(2340)': boson_bdensity_integ, 'Lambda(2350)': fermion_bdensity_integ,
    'Delta(2420)': fermion_bdensity_integ, 'N(2600)': fermion_bdensity_integ,
}
    

def pressure(T: float, muB: float, hadron: str) -> float:
    
    M = MI[hadron]
    B = BI[hadron]
    D = DI[hadron]

    integral, error = scipy.integrate.quad(
        pressure_hash[hadron], 0.0, math.inf, args = (T, muB, M, B)
    )

    return (D/(6.0*(math.pi**2)))*integral


def sdensity(T: float, muB: float, hadron: str) -> float:
    
    M = MI[hadron]
    B = BI[hadron]
    D = DI[hadron]

    integral, error = scipy.integrate.quad(
        sdensity_hash[hadron], 0.0, math.inf, args = (T, muB, M, B)
    )

    return (D/(2.0*(math.pi**2)))*integral


def bdensity(T: float, muB: float, hadron: str) -> float:
    
    M = MI[hadron]
    B = BI[hadron]
    D = DI[hadron]

    if B == 0.0:
        return 0.0
    else:
        integral, error = scipy.integrate.quad(
            bdensity_hash[hadron], 0.0, math.inf, args = (T, muB, M, B)
        )
        return B*(D/(2.0*(math.pi**2)))*integral


def pressure_multi(
    T: float, muB: float, hadrons="all"
):
    partial = list()
    if hadrons == "all":
        for hadron in MI:
            partial.append(pressure(T, muB, hadron))
    else:
        for hadron in hadrons:
            partial.append(pressure(T, muB, hadron))
    return math.fsum(partial), (*partial,)


def sdensity_multi(
    T: float, muB: float, hadrons="all"
):
    partial = list()
    if hadrons == "all":
        for hadron in MI:
            partial.append(sdensity(T, muB, hadron))
    else:
        for hadron in hadrons:
            partial.append(sdensity(T, muB, hadron))
    return math.fsum(partial), (*partial,)


def bdensity_multi(
    T: float, muB: float, hadrons="all"
):
    partial = list()
    if hadrons == "all":
        for hadron in MI:
            partial.append(bdensity(T, muB, hadron))
    else:
        for hadron in hadrons:
            partial.append(bdensity(T, muB, hadron))
    return math.fsum(partial), (*partial,)