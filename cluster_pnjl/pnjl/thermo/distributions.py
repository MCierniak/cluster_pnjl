import math

def f_fermion_singlet(y_val, y_status) -> float:
    #
    return 1.0 / (y_val + 1.0)

def f_baryon_singlet(y_val, y_status) -> float:
    if y_status == 2:
        return math.inf
    else:
        return 1.0 / (y_val - 1.0)

def f_fermion_triplet(
    Phi : complex, Phib : complex, 
    y_1_val, y_1_status,
    y_2_val, y_2_status,
    y_3_val, y_3_status,
    ) -> complex:
    part1 = 3.0 * Phib / (y_plus_1_val + 3.0 * Phib + 3.0 * Phi * (1.0 / y_plus_1_val) + (1.0 / y_plus_2_val)) if 1 not in [y_plus_1_status, y_plus_2_status] else complex(0.0, 0.0)
    part2 = 6.0 * Phi / (y_plus_2_val + 3.0 * Phib * y_plus_1_val + 3.0 * Phi + (1.0 / y_plus_1_val)) if 1 not in [y_plus_1_status] else complex(0.0, 0.0)
    part3 = 3.0 / (y_plus_3_val + 3.0 * Phib * y_plus_2_val + 3.0 * Phi * y_plus_1_val + 1.0)
    return part1 + part2 + part3

def f_fermion_antitriplet(
    Phi : complex, Phib : complex, 
    y_1_val, y_1_status,
    y_2_val, y_2_status,
    y_3_val, y_3_status,
    ) -> complex:
    #
    return f_fermion_triplet(Phib, Phi, y_1_val, y_1_status, y_2_val, y_2_status, y_3_val, y_3_status)

def f_baryon_triplet(
    Phi : complex, Phib : complex, 
    y_1_val, y_1_status,
    y_2_val, y_2_status,
    y_3_val, y_3_status,
    ) -> complex:
    part1 = 3.0 * Phib / (y_plus_1_val - 3.0 * Phib + 3.0 * Phi * (1.0 / y_plus_1_val) - (1.0 / y_plus_2_val)) if 1 not in [y_plus_1_status, y_plus_2_status] else complex(0.0, 0.0)
    part2 = -6.0 * Phi / (y_plus_2_val - 3.0 * Phib * y_plus_1_val + 3.0 * Phi - (1.0 / y_plus_1_val)) if 1 not in [y_plus_1_status] else complex(0.0, 0.0)
    part3 = 3.0 / (y_plus_3_val - 3.0 * Phib * y_plus_2_val + 3.0 * Phi * y_plus_1_val - 1.0)
    return part1 + part2 + part3

def f_baryon_antitriplet(
    Phi : complex, Phib : complex, 
    y_1_val, y_1_status,
    y_2_val, y_2_status,
    y_3_val, y_3_status,
    ) -> complex:
    #
    return f_baryon_triplet(Phib, Phi, y_1_val, y_1_status, y_2_val, y_2_status, y_3_val, y_3_status)