import numpy as np

def prox_H(B, C, delta1, delta2, Z1, Z2, lambda_N, WGTs):

    deltas = delta1 + delta2
    b_delta1 = B - Z1 / delta1
    b_delta2 = C - Z2 / delta2
    b_delta = (delta1 * b_delta1 + delta2 * b_delta2) / deltas
    d_new = np.sign(b_delta) * np.maximum(abs(b_delta) - lambda_N * WGTs / deltas, 0)
    return d_new