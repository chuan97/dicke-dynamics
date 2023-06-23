import numpy as np

from scipy.linalg import eigh

import exact
import level_statistics as level_stat

Ns = [40]
N_cutoffs = [150]

wc = 1
wz = 1 * wc
lams = [2.0]

for N in Ns:
    for N_c in N_cutoffs:
        for lam in lams:
            # regular Dicke
            P = level_stat.Peven(N/2, N_c)
            H = exact.dicke(wz, wc, lam, N/2, N_c)
            projected_H = P @ H @ P
            vals = eigh(projected_H, eigvals_only=True)
            vals = level_stat.trim_vals(vals, N, N_c)
            
            np.save(f'data/spectra/sprectrum_dicke_{N}_{N_c}_{wc}_{wz}_{lam}', vals)
            
            # Dicke with P2
            P = level_stat.Peven(N/2, N_c)
            H = exact.dicke_P2(wz, wc, lam, N/2, N_c)
            projected_H = P @ H @ P
            vals = eigh(projected_H, eigvals_only=True)
            vals = level_stat.trim_vals(vals, N, N_c)
            
            np.save(f'data/spectra/sprectrum_dickeP2_{N}_{N_c}_{wc}_{wz}_{lam}', vals)