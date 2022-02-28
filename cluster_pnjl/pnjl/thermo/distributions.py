import math

def f_fermion_singlet(y_val = 0.0, y_status = 4) -> float:
    if y_status == 4:
        raise RuntimeError("Error in pnj.thermo.distributions.f_fermion_singlet, y value not passed...")
    else:
        return 1.0 / (y_val + 1.0)

def f_baryon_singlet(y_val = 0.0, y_status = 4) -> float:
    if y_status == 4:
        raise RuntimeError("Error in pnj.thermo.distributions.f_baryon_singlet, y value not passed...")
    elif y_status == 2:
        return math.inf
    else:
        return 1.0 / (y_val - 1.0)

def f_fermion_triplet(
    Phi : complex, Phib : complex, 
    y_1_val = 0.0, y_1_status = 4,
    y_2_val = 0.0, y_2_status = 4,
    y_3_val = 0.0, y_3_status = 4,
    ) -> complex:
    if 4 in [y_1_status, y_2_status, y_3_status]:
        raise RuntimeError("Error in pnj.thermo.distributions.f_fermion_triplet, y value not passed...")

    y_m1_val = 1.0 / y_1_val if not y_1_status == 1 else math.inf
    y_m2_val = 1.0 / y_2_val if not y_2_status == 1 else math.inf

    den1 = y_1_val + 3.0 * Phib + 3.0 * Phi * y_m1_val + y_m2_val
    den2 = y_2_val + 3.0 * Phib * y_1_val + 3.0 * Phi + y_m1_val
    den3 = y_3_val + 3.0 * Phib * y_2_val + 3.0 * Phi * y_1_val + 1.0

    part1 = 3.0 * Phib / den1 if not math.fabs(den1.real) == math.fabs(math.inf) and not math.fabs(den1.imag) == math.fabs(math.inf) else complex(0.0, 0.0)
    part2 = 6.0 * Phi / den2 if not math.fabs(den2.real) == math.fabs(math.inf) and not math.fabs(den2.imag) == math.fabs(math.inf) else complex(0.0, 0.0)
    part3 = 3.0 / den3 if not math.fabs(den3.real) == math.fabs(math.inf) and not math.fabs(den3.imag) == math.fabs(math.inf) else complex(0.0, 0.0)

    #print("y_1_val", y_1_val)
    #print("y_2_val", y_2_val)
    #print("y_3_val", y_3_val)
    #print("y_m1_val", y_m1_val)
    #print("y_m2_val", y_m2_val)
    #print("den1", den1)
    #print("den2", den2)
    #print("den3", den3)
    #print("part1", part1)
    #print("part2", part2)
    #print("part3", part3)
    #input()

    return part1 + part2 + part3

def f_fermion_antitriplet(
    Phi : complex, Phib : complex, 
    y_1_val = 0.0, y_1_status = 4,
    y_2_val = 0.0, y_2_status = 4,
    y_3_val = 0.0, y_3_status = 4,
    ) -> complex:
    #
    if 4 in [y_1_status, y_2_status, y_3_status]:
        raise RuntimeError("Error in pnj.thermo.distributions.f_fermion_antitriplet, y value not passed...")
    return f_fermion_triplet(Phib, Phi, y_1_val = y_1_val, y_1_status = y_1_status, y_2_val = y_2_val, y_2_status = y_2_status, y_3_val = y_3_val, y_3_status = y_3_status)

def f_baryon_triplet(
    Phi : complex, Phib : complex, 
    y_1_val = 0.0, y_1_status = 4,
    y_2_val = 0.0, y_2_status = 4,
    y_3_val = 0.0, y_3_status = 4,
    ) -> complex:
    if 4 in [y_1_status, y_2_status, y_3_status]:
        raise RuntimeError("Error in pnj.thermo.distributions.f_baryon_triplet, y value not passed...")

    y_m1_val = 1.0 / y_1_val if not y_1_status == 1 else math.inf
    y_m2_val = 1.0 / y_2_val if not y_2_status == 1 else math.inf

    den1 = y_1_val - 3.0 * Phib + 3.0 * Phi * (1.0 / y_1_val) - (1.0 / y_2_val)
    den2 = y_2_val - 3.0 * Phib * y_1_val + 3.0 * Phi - (1.0 / y_1_val)
    den3 = y_3_val - 3.0 * Phib * y_2_val + 3.0 * Phi * y_1_val - 1.0

    part1 = 3.0 * Phib / den1 if not math.fabs(den1.real) == math.fabs(math.inf) and not math.fabs(den1.imag) == math.fabs(math.inf) else complex(0.0, 0.0)
    part2 = -6.0 * Phi / den2 if not math.fabs(den2.real) == math.fabs(math.inf) and not math.fabs(den2.imag) == math.fabs(math.inf) else complex(0.0, 0.0)
    part3 = 3.0 / den3 if not math.fabs(den3.real) == math.fabs(math.inf) and not math.fabs(den3.imag) == math.fabs(math.inf) else complex(0.0, 0.0)
    return part1 + part2 + part3

def f_baryon_antitriplet(
    Phi : complex, Phib : complex, 
    y_1_val = 0.0, y_1_status = 4,
    y_2_val = 0.0, y_2_status = 4,
    y_3_val = 0.0, y_3_status = 4,
    ) -> complex:
    if 4 in [y_1_status, y_2_status, y_3_status]:
        raise RuntimeError("Error in pnj.thermo.distributions.f_baryon_antitriplet, y value not passed...")
    return f_baryon_triplet(Phib, Phi, y_1_val, y_1_status, y_2_val, y_2_status, y_3_val, y_3_status)