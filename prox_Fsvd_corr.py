import numpy as np
def prox_fsvd_corr(SVD_Z, D_k, Z_k, delta):

    Sdiag = SVD_Z["Sdiag"]
    SdiagSq = SVD_Z["Sdiagsq"]
    Vt = SVD_Z["Vt"]
    StUty= SVD_Z["StUtY"]

    p = Vt.shape[0]

    D = D_k + Z_k / delta
    VtD = Vt @ D
    diags = SdiagSq + delta

    B_new = (Vt.T /diags) @ (StUty + delta * VtD)

    return B_new

