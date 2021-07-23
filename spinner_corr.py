import numpy as np
from sol_options import Options
from Spinner_both_corr import calculate_spinner_both_lambdas_corr
import seaborn as sns
from scipy.linalg import block_diag
def spinner_correlation(Y,Z,lambdaN, lambdaL):



    # Checking: lambdas non-negative
    if np.any(np.array([lambdaL,lambdaN]) < 0):
        raise ValueError

    p = Z.shape[1]
    W = np.ones((p, p)) - np.eye(p, p)

    U, Sdiag, Vt = np.linalg.svd(Z)
    S = np.diag(Sdiag)
    #print(f"U shape{U.shape}, S shape {S.shape}, Vt shape {Vt.shape}")
    np.allclose(Z, U[:, :Sdiag.shape[0]] @ S @ Vt[:Sdiag.shape[0], :])

    #########
    ######### SVD objects
    #########

    U = U[:, :S.shape[0]]  # czy tutaj zapisuje to co nie mnożymy przez 0 ?
    middle_product = U[:, :S.shape[0]] @ S  # złożenie pierwszych 2 macierzy
    Vt = Vt[:middle_product.shape[0], :]

    SVD_Z = {}
    SVD_Z["Sdiag"] = Sdiag
    SVD_Z["Sdiagsq"] = Sdiag**2
    SVD_Z["Vt"] = Vt
    SVD_Z["StUtY"] = S.T @ U.T @ Y

    ## cases

    ## Cases
    solverType = [x > 0 for x in [lambdaN, lambdaL]]
    solverType = solverType[0] + 2 * solverType[1] + 1

    if solverType != 4:
        print("One of lambda's values is equal to 0")
    else:
        out = calculate_spinner_both_lambdas_corr(Y, SVD_Z, lambdaN, lambdaL, W)

    estim = out["B"]
    residuals = ((Y - estim)**2).sum()
    out["residuals"] = residuals
    #print(residuals)

    return out

if __name__ == "__main__":
    p = 60
    n = 100
    B1 = 1 * np.ones((15, 15))
    B2 = -1 * np.ones((12, 12))
    B3 = 1 * np.ones((10, 10))
    s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
    left_square = p - s_nods - 18  #
    BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                   np.zeros((left_square, left_square)))
    np.fill_diagonal(BB, 15)
    np.all(np.linalg.eigvals(BB) > 0)
    sns.heatmap(BB, center=0, vmin=-2, vmax=2)

    X = np.zeros((n, p))
    mean = np.zeros(p)
    cov = BB
    np.random.seed(2020)
    for row in range(X.shape[0]):
        X[row, :] = np.random.multivariate_normal(mean, cov, 1)

    #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    Y = X.T @ X
    Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    #Y = Y - np.diag(Y)
    np.cov(X.T, bias=True) == Y
    sns.heatmap(np.cov(X.T) , center=0, vmin=-1, vmax=1)
    np.fill_diagonal(Y,0)
    sns.heatmap(Y, center=0, vmin=-1, vmax=1)
    Z = np.identity(Y.shape[0])
    lambdaN = 2
    lambdaL = 2

    Y_magic = BB + np.random.randn(p,p)
    np.fill_diagonal(Y_magic, 0)
    out = spinner_correlation(Y_magic, Z, 3, 0.15)
    sns.heatmap(out["B"], center=0, vmin=-3, vmax=3)

    Y

    out = spinner_correlation(Y,Z,5, 8)

    sns.heatmap(out["B"], center=0, vmin=-20, vmax=20)

    sns.heatmap(BB, center=0, vmin=-2, vmax=2)
