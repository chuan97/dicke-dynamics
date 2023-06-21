import numpy as np
from scipy.linalg import kron

def Peven(S, n_cutoff):
    size = int((2*S + 1) * (n_cutoff + 1))
    Peven = np.zeros((size, size))
    for i, mz in enumerate(np.arange(-S, S + 1)):
        for n in np.arange(n_cutoff + 1):
            if (mz + S + n) % 2 == 0:
                Peven += kron(Pmz(i, S), Pn(n, n_cutoff))
                
    return Peven

def Pmz(i, S):
    diag = np.zeros(int(2 * S + 1))
    diag[i] = 1
    return np.diag(diag)

def Pn(n, n_cutoff):
    diag = np.zeros(n_cutoff + 1)
    diag[n] = 1
    return np.diag(diag)

def moving_average(a, window):
    avg = np.empty(len(a))
    for i, _ in enumerate(a):
        low_lim = max(0, i-window//2)
        up_lim = min(len(a), i+window//2)
        avg[i] = np.average(a[low_lim:up_lim])
        
    return avg

def trim_hamiltonian(H):
    # has a problem: for practical sizes the recursion limit is reached
    for i in range(H.shape[0]):
        if np.all(H[i] == 0):
            row_removed = np.delete(H, i, axis=0)
            rowandcolumn_removed = np.delete(row_removed, i, axis=1)
            return trim_hamiltonian(rowandcolumn_removed)
            
    return H

def trim_vals(vals, N, N_photons_cutoff):
    assert len(np.where(np.abs(vals) < 1e-10)[0]) >= (N+1)*(N_photons_cutoff+1)//2
    return np.delete(vals, np.where(np.abs(vals) < 1e-10)[0][:(N+1)*(N_photons_cutoff+1)//2])

def wigner_surmise_degree(renormalized_spacings, bins=500):
    s0 = 0.472913
    counts, bin_edges = np.histogram(renormalized_spacings, bins=bins, range=(0, 10), density=True) # important to fix range, otherwise each run changes the bin size
    ds = bin_edges[1] - bin_edges[0]
    s = np.arange(0, s0, ds)
    
    wigner = lambda s: 0.5*np.pi*s * np.exp(-0.25*np.pi*s**2)
    #print(ds, np.sum(counts[np.where(bin_edges[1:] < s0)[0]]) * ds, np.sum(wigner(s)) * ds, np.sum(np.exp(-s)) * ds)
    
    return np.abs((np.sum(counts[np.where(bin_edges[1:] < s0)[0]]) - np.sum(wigner(s))) / (np.sum(np.exp(-s)) - np.sum(wigner(s))))
