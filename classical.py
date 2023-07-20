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

def Dicke_classical_eqsmot_second_order(t, v, wz, wc, lam):
    dv = np.empty(v.shape)
    dv[0] = -wz*v[1] # mx
    dv[1] = wz*v[0] - 2*lam*v[18] # my
    dv[2] = 2*lam*v[13] # mz
    dv[3] = -wc*v[4] # x
    dv[4] = wc*v[3] + 2*lam*v[0] # p
    dv[5] = -wz*v[6] - wz*v[10] # mx mx 
    dv[6] = -wz*v[11] + wz*v[5] + 4*lam*v[0]*v[2]*v[3] - 2*lam*v[0]*v[18] - 2*lam*v[2]*v[8] - 2*lam*v[3]*v[7] # mx my
    dv[7] = -wz*v[12] - 4*lam*v[0]*v[1]*v[3] + 2*lam*v[0]*v[13] + 2*lam*v[1]*v[8] + 2*lam*v[3]*v[6] # mx mz
    dv[8] = -wz*v[13] - wc*v[9] # mx x
    dv[9] = -wz*v[14] + wc*v[8] + 2*lam*v[5] # mx p
    dv[10] = -wz*v[11] + wz*v[5] + 4*lam*v[0]*v[2]*v[3] - 2*lam*v[2]*v[8] - 2*lam*v[0]*v[18] - 2*lam*v[3]*v[15] # my mx
    dv[11] = wz*v[10] + wz*v[6] + 8*lam*v[1]*v[2]*v[3] - 4*lam*v[1]*v[18] - 4*lam*v[2]*v[13] - 2*lam*v[3]*v[12] - 2*lam*v[3]*v[16] # my my
    dv[12] = wz*v[7] + 4*lam*(v[2]**2)*v[3] - 4*lam*v[2]*v[18] - 2*lam*v[3]*v[17] - 4*lam*(v[1]**2)*v[3] + 4*lam*v[1]*v[13] + 2*lam*v[3]*v[11] # my mz
    dv[13] = wz*v[8] - wc*v[14] + 4*lam*v[2]*(v[3]**2) - 2*lam*v[2]*v[20] - 4*lam*v[3]*v[18] # my x
    dv[14] = wz*v[9] + wc*v[13] + 4*lam*v[2]*v[3]*v[4] - 2*lam*v[2]*v[22] - 2*lam*v[3]*v[19] - 2*lam*v[4]*v[18] + 2*lam*v[6] # my p
    dv[15] = -wz*v[16] - 4*lam*v[0]*v[1]*v[3] + 2*lam*v[1]*v[8] + 2*lam*v[0]*v[13] + 2*lam*v[3]*v[10] # mz mx
    dv[16] = wz*v[15] + 4*lam*(v[2]**2)*v[3] - 4*lam*v[2]*v[18] - 2*lam*v[3]*v[17] - 4*lam*(v[1]**2)*v[3] + 4*lam*v[1]*v[13] + 2*lam*v[3]*v[11] # mz my
    dv[17] = -8*lam*v[1]*v[2]*v[3] + 4*lam*v[2]*v[13] + 4*lam*v[1]*v[18] + 2*lam*v[3]*v[16] + 2*lam*v[3]*v[12] # mz mz
    dv[18] = -wc*v[19] - 4*lam*v[1]*(v[3]**2) + 2*lam*v[1]*v[20] + 4*lam*v[3]*v[13] # mz x
    dv[19] = wc*v[18] - 4*lam*v[1]*v[3]*v[4] + 2*lam*v[1]*v[22] + 2*lam*v[3]*v[14] + 2*lam*v[4]*v[13] + 2*lam*v[7] # mz p
    dv[20] = -wc*v[21] - wc*v[22] # x x
    dv[21] = -wc*v[23] + wc*v[20] + 2*lam*v[8] # x p
    dv[22] = -wc*v[23] + wc*v[20] + 2*lam*v[8] # p x
    dv[23] = wc*v[21] + wc*v[22] + 4*lam*v[9] # p p
        
    return dv

def Dicke_classical_eqsmot_second_order_alt(t, v, wz, wc, lam):
    dv = np.empty(v.shape)
    dv[0] = -wz*v[1] # mx
    dv[1] = wz*v[0] - 2*lam*v[18] # my
    dv[2] = 2*lam*v[13] # mz
    dv[3] = -wc*v[4] # x
    dv[4] = wc*v[3] + 2*lam*v[0] # p
    dv[5] = -wz*v[6] - wz*v[10] # mx mx 
    dv[6] = -wz*v[11] + wz*v[5] + 4*lam*v[0]*v[2]*v[3] - 2*lam*v[0]*v[18] - 2*lam*v[2]*v[8] - 2*lam*v[3]*v[7] # mx my
    dv[7] = -wz*v[12] - 4*lam*v[0]*v[1]*v[3] + 2*lam*v[0]*v[13] + 2*lam*v[1]*v[8] + 2*lam*v[3]*v[6] # mx mz
    dv[8] = -wz*v[13] - wc*v[9] # mx x
    dv[9] = -wz*v[14] + wc*v[8] + 2*lam*v[5] # mx p
    dv[10] = -wz*v[11] + wz*v[5] + 4*lam*v[0]*v[2]*v[3] - 2*lam*v[2]*v[8] - 2*lam*v[0]*v[18] - 2*lam*v[3]*v[15] # my mx
    dv[11] = wz*v[10] + wz*v[6] + 8*lam*v[1]*v[2]*v[3] - 4*lam*v[1]*v[18] - 4*lam*v[2]*v[13] - 2*lam*v[3]*v[12] - 2*lam*v[3]*v[16] # my my
    dv[12] = wz*v[7] + 4*lam*(v[2]**2)*v[3] - 4*lam*v[2]*v[18] - 2*lam*v[3]*v[17] - 4*lam*(v[1]**2)*v[3] + 4*lam*v[1]*v[13] + 2*lam*v[3]*v[11] # my mz
    dv[13] = wz*v[8] - wc*v[14] + 4*lam*v[2]*(v[3]**2) - 2*lam*v[2]*v[20] - 4*lam*v[3]*v[18] # my x
    dv[14] = wz*v[9] + wc*v[13] + 4*lam*v[2]*v[3]*v[4] - 2*lam*v[2]*v[21] - 2*lam*v[3]*v[19] - 2*lam*v[4]*v[18] + 2*lam*v[10] # my p
    dv[15] = -wz*v[16] - 4*lam*v[0]*v[1]*v[3] + 2*lam*v[1]*v[8] + 2*lam*v[0]*v[13] + 2*lam*v[3]*v[10] # mz mx
    dv[16] = wz*v[15] + 4*lam*(v[2]**2)*v[3] - 4*lam*v[2]*v[18] - 2*lam*v[3]*v[17] - 4*lam*(v[1]**2)*v[3] + 4*lam*v[1]*v[13] + 2*lam*v[3]*v[11] # mz my
    dv[17] = -8*lam*v[1]*v[2]*v[3] + 4*lam*v[2]*v[13] + 4*lam*v[1]*v[18] + 2*lam*v[3]*v[16] + 2*lam*v[3]*v[12] # mz mz
    dv[18] = -wc*v[19] - 4*lam*v[1]*(v[3]**2) + 2*lam*v[1]*v[20] + 4*lam*v[3]*v[13] # mz x
    dv[19] = wc*v[18] - 4*lam*v[1]*v[3]*v[4] + 2*lam*v[1]*v[21] + 2*lam*v[3]*v[14] + 2*lam*v[4]*v[13] + 2*lam*v[15] # mz p
    dv[20] = -wc*v[21] - wc*v[22] # x x
    dv[21] = -wc*v[23] + wc*v[20] + 2*lam*v[8] # x p
    dv[22] = -wc*v[23] + wc*v[20] + 2*lam*v[8] # p x
    dv[23] = wc*v[21] + wc*v[22] + 4*lam*v[9] # p p
        
    return dv

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