import numpy as np
import math

from collections import namedtuple

from scipy.linalg import kron, expm
from scipy.sparse.linalg import expm as expms
from scipy.sparse import kron as krons
from scipy.sparse import diags, eye, csr_matrix

def spin_operators(S, *, to_dense_array=False, format=None, dtype=np.float_):
    Sz = diags([m for m in np.arange(-S, S + 1)], format=format, dtype=dtype)
    Sp = diags([math.sqrt(S * (S + 1) - m * (m + 1)) for m in np.arange(-S, S)],
               offsets=-1,
               format=format,
               dtype=dtype
               )
    Sm = Sp.T
    Seye = eye(2 * S + 1, format=format, dtype=dtype)

    Spin_operators = namedtuple('Spin_operators', 'Sz Sp Sm Seye')
    ops = Spin_operators(Sz, Sp, Sm, Seye)
    if to_dense_array:
        ops = Spin_operators(*[o.toarray() for o in ops])

    return ops


def boson_operators(N_photons_cutoff, *, to_dense_array=False, format=None, dtype=np.float_):
    a = diags([math.sqrt(n) for n in range(1, N_photons_cutoff + 1)],
              offsets=1,
              format=format,
              dtype=dtype
              )
    ad = a.T
    beye = eye(N_photons_cutoff + 1, format=format, dtype=dtype)

    Boson_operators = namedtuple('Boson_operators', 'a ad beye')
    ops = Boson_operators(a, ad, beye)
    if to_dense_array:
        ops = Boson_operators(*[o.toarray() for o in ops])

    return ops


def dicke(ws, wc, lam, S, N_photons_cutoff, to_dense_array=True):
    kron_ = kron if to_dense_array else krons
    g = lam / np.sqrt(2*S)
    Sz, Sp, Sm, Seye = spin_operators(S, to_dense_array=to_dense_array)
    Sx = 0.5 * (Sp + Sm)
    a, ad, beye = boson_operators(N_photons_cutoff, to_dense_array=to_dense_array)
    H = ws*kron_(Sz, beye) + wc*kron_(Seye, ad @ a) + 2*g*kron_(Sx, a + ad)
    
    return H

def dicke_P2(ws, wc, lam, S, N_photons_cutoff):
    g = lam / np.sqrt(2*S)
    Sz, Sp, Sm, Seye = spin_operators(S, to_dense_array=True)
    Sx = 0.5 * (Sp + Sm)
    a, ad, beye = boson_operators(N_photons_cutoff, to_dense_array=True)

    H = ws*kron(Sz, beye) + wc*kron(Seye, ad @ a) + 2*g*kron(Sx, a + ad) + 4*g**2*kron(Sx@Sx, beye)
    return H

def coshm(a, to_dense_array=True):
    return simexp(a, 1, to_dense_array=to_dense_array)

def sinhm(a, to_dense_array=True):
    return simexp(a, -1, to_dense_array=to_dense_array)

def simexp(a, pm, to_dense_array=True):
    expm_ = expm if to_dense_array else expms
    return 0.5 * (expm_(a) + pm*expm_(-a))

def Dicke_polaron(ws, wc, lam, eps, S, N_photons_cutoff, to_dense_array=True):
    g = lam / np.sqrt(2*S)
    kron_ = kron if to_dense_array else krons
    Sz, Sp, Sm, Seye = spin_operators(S, to_dense_array=to_dense_array)
    Sx = 0.5 * (Sp + Sm)
    a, ad, beye = boson_operators(N_photons_cutoff, to_dense_array=to_dense_array)
    
    H = ws * (kron_(Sz, coshm(2 * g / wc * (ad - a), to_dense_array=to_dense_array)) - kron_(0.5 * (Sp - Sm), sinhm(2 * g / wc * (ad - a), to_dense_array=to_dense_array)))\
        + wc * kron_(Seye, ad @ a) - 4 * g**2 / wc * kron_(Sx @ Sx, beye) - eps * kron_(Sx, beye) 

    return H

def polaritons(wz, wc, lam):
    def para(wz, wc, lam):
        innerroot = np.sqrt(wz**4 + wc**4 - 2*wz**2*wc**2 + 16*lam**2*wz*wc)
        return np.sqrt(0.5 * (wz**2 + wc**2 - innerroot)), np.sqrt(0.5 * (wz**2 + wc**2 + innerroot))
    
    def ferro(wz, wc, lam):
        mu = wz * wc / (4 * lam**2)
        g = lam * mu * np.sqrt(2 / (1 + mu))
        wztilde = wz * (1 + mu) / (2 * mu)
        eps = wz * (1 - mu) * (3 + mu) / (8 * mu * (1 + mu))
        
        innerroot = np.sqrt(wztilde**4 + wc**4 - 2*wztilde**2*wc**2 + 16*g**2*wztilde*wc + 4*(eps**2*wztilde**2 + eps*wztilde**3 - wc**2*eps*wztilde))
        return np.sqrt(0.5 * (wztilde**2 + wc**2 + 2*eps*wztilde - innerroot)), np.sqrt(0.5 * (wztilde**2 + wc**2 + 2*eps*wztilde + innerroot))
    
    if 4 * lam**2 < wz * wc:
        # paramagnetic phase
        return para(wz, wc, lam)
    else:
        # ferromagnetic phase
        return ferro(wz, wc, lam)
