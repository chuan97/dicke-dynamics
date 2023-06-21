import numpy as np

def Dicke_eqsmot(t, mxp, wz, wc, lam, N):
    m = mxp[:-2]
    xp = mxp[-2:]
        
    Bext = np.array([0, 0, wz])
    Brms = np.array([2 * lam / np.sqrt(N), 0, 0])
    Beff = Bext + Brms*xp[0]
    dm = -np.cross(m, Beff) 

    dxp = np.empty(xp.shape)
    dxp[0] = -wc*xp[1]
    dxp[1] = wc*xp[0] + N*np.dot(Brms, m)
    return np.append(dm, dxp)

def Dicke_explicit_eqsmot(t, mxp, wz, wc, lam, N):
    dmxp = np.empty(mxp.shape)
    dmxp[0] = -wz*mxp[1]
    dmxp[1] = wz*mxp[0] - 2*lam/np.sqrt(N)*mxp[3]*mxp[2]
    dmxp[2] = 2*lam/np.sqrt(N)*mxp[3]*mxp[1]
    dmxp[3] = -wc*mxp[4]
    dmxp[4] = wc*mxp[3] + N*2*lam/np.sqrt(N)*mxp[0]
    
    return dmxp

def Dicke_explicit_cavdis_eqsmot(t, mxp, wz, wc, lam, kappa, N):
    dmxp = np.empty(mxp.shape)
    dmxp[0] = -wz*mxp[1]
    dmxp[1] = wz*mxp[0] - 2*lam/np.sqrt(N)*mxp[3]*mxp[2]
    dmxp[2] = 2*lam/np.sqrt(N)*mxp[3]*mxp[1]
    dmxp[3] = -wc*mxp[4] - kappa*mxp[3]
    dmxp[4] = wc*mxp[3] + N*2*lam/np.sqrt(N)*mxp[0] - kappa*mxp[4]
    
    return dmxp

def Dicke_classical_eqsmot(t, mxp, wz, wc, lam, kappa):
    dmxp = np.empty(mxp.shape)
    dmxp[0] = -wz*mxp[1]
    dmxp[1] = wz*mxp[0] - 2*lam*mxp[3]*mxp[2]
    dmxp[2] = 2*lam*mxp[3]*mxp[1]
    dmxp[3] = -wc*mxp[4] - kappa*mxp[3]
    dmxp[4] = wc*mxp[3] + 2*lam*mxp[0] - kappa*mxp[4]
    
    return dmxp

def Dicke_P2_cavdis_eqsmot(t, mxp, wz, wc, lam, kappa, N):
    dmxp = np.empty(mxp.shape)
    dmxp[0] = -wz*mxp[1]
    dmxp[1] = wz*mxp[0] - 2*lam/np.sqrt(N)*mxp[3]*mxp[2] - 4*lam**2/wc*mxp[0]*mxp[2]
    dmxp[2] = 2*lam/np.sqrt(N)*mxp[3]*mxp[1] + 4*lam**2/wc*mxp[0]*mxp[1]
    dmxp[3] = -wc*mxp[4] - kappa*mxp[3]
    dmxp[4] = wc*mxp[3] + N*2*lam/np.sqrt(N)*mxp[0] - kappa*mxp[4]
    
    return dmxp

def Dicke_P2_eqsmot(t, mxp, wz, wc, lam, N):
    dmxp = np.empty(mxp.shape)
    dmxp[0] = -wz*mxp[1]
    dmxp[1] = wz*mxp[0] - 2*lam/np.sqrt(N)*mxp[3]*mxp[2] - 4*lam**2/wc*mxp[0]*mxp[2]
    dmxp[2] = 2*lam/np.sqrt(N)*mxp[3]*mxp[1] + 4*lam**2/wc*mxp[0]*mxp[1]
    dmxp[3] = -wc*mxp[4] 
    dmxp[4] = wc*mxp[3] + N*2*lam/np.sqrt(N)*mxp[0]
    
    return dmxp

def Dicke_P2_classical_eqsmot(t, mxp, wz, wc, lam, kappa):
    dmxp = np.empty(mxp.shape)
    dmxp[0] = -wz*mxp[1]
    dmxp[1] = wz*mxp[0] - 2*lam*mxp[3]*mxp[2] - 4*lam**2/wc*mxp[0]*mxp[2]
    dmxp[2] = 2*lam*mxp[3]*mxp[1] + 4*lam**2/wc*mxp[0]*mxp[1]
    dmxp[3] = -wc*mxp[4] - kappa*mxp[3]
    dmxp[4] = wc*mxp[3] + 2*lam*mxp[0] - kappa*mxp[4]
    
    return dmxp

def LMG_eqsmot(t, m, wz, wc, lam, N):
    dm = np.empty(m.shape)
    dm[0] = -wz*m[1]
    dm[1] = wz*m[0] + 4*lam**2/wc*m[0]*m[2]
    dm[2] = -4*lam**2/wc*m[0]*m[1]
    
    return dm 