
import numpy as np

def prox_G(D, Z2, delta, lambda_N):
    d_delta = D + Z2 / delta
    d_delta = (d_delta + d_delta.T) / 2  
    U, S, Vt = np.linalg.svd(d_delta)  # Ddelta = U*diag(S)*Vt
    diagS = np.diag(S)
    if np.allclose(d_delta, U @ diagS @ Vt) != True:
        print("ProxG SVD not made properly")
    Stsh = np.diag(np.sign(S) * np.maximum(np.abs(S) - lambda_N / delta, 0))  # soft tresholoding of singular values
    c_new = U @ Stsh @ Vt
    return c_new