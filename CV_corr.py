
import  numpy as np
import seaborn as sns
from scipy.linalg import block_diag
import math
import matplotlib.pyplot as plt
import sys
from spinner_corr import spinner_correlation
from tqdm import tqdm



def CV_calculate(X):
    n, p = X.shape
    X = (X - np.mean(X, axis=0))
    Y_checked = (X.T @X)/X.shape[0]
    #print(f"Error dla czystej macierzy covariancji wynosi {((Y_checked - BB)**2).sum()}")

    #put_to_diagonal = np.diagonal(Y_checked).copy()
    np.fill_diagonal(Y_checked,0)
    #sns.heatmap(Y_checked , center=0, vmin=-20, vmax=20)


    grid_lengthN = 15
    grid_lengthL = 15
    kfolds = 5

    ### CV options
    initi_lambda = 1
    zero_search_ratio = 100
    max_lambd_acc = 1e-2

    ### CV indices
    minElemsN = math.floor(n / kfolds)
    one_to_K_folds = np.arange(1,kfolds+1,1)
    groups_idxs = np.tile(one_to_K_folds, (minElemsN, 1)) # 120x5 [1,2,3,4,5]
    groups_idxs = groups_idxs.T.reshape((-1,))
    np.random.seed(2021)
    random_sample = np.random.choice(n, n, replace=False)# ze zwracaniem czy bez ??
    groups_idxs = groups_idxs[random_sample]

# unique, counts = np.unique(groups_idxs, return_counts=True)
# dict(zip(unique, counts))

# Set up W and Z
    Z = np.identity(p)
    W = np.ones((p, p)) - np.eye(p, p)

    # FINDING the maximal lambda L
    c_lambdaL = initi_lambda
    vals_lamd_L = []
    counterr1 = 1
    stop = 0
    # finding lambda_L for which matrix of zeros is obtained
    while stop==0:
        out = spinner_correlation(Y_checked,Z,1e-3,c_lambdaL )
        if np.sqrt(((W * out["B"]) ** 2).sum()) < 1e-16:
            stop = 1
        vals_lamd_L.append(c_lambdaL)
        c_lambdaL = zero_search_ratio * c_lambdaL
        counterr1 += 1
        #print("OTO MÓJ COUNTER", counterr1)
    if len(vals_lamd_L) == 1:
        lam_L1 = 0
        lam_L2 = vals_lamd_L[0]
    else:
        lam_L1 = vals_lamd_L[-2]
        lam_L2 = vals_lamd_L[-1]

    ### finding narrow interval form amximal lambda L
    counterr2 = 1
    vals_lamd_L_max = []

    while True:
        c_lam_L_max_new = (lam_L1 + lam_L2) / 2
        out_new_o =  spinner_correlation(Y_checked,Z,1e-3,c_lam_L_max_new )
        #print("OTO MÓJ COUNTER", counterr2)

        if np.linalg.norm(out_new_o["B"], "fro") < 1e-16:
            lam_L2 = c_lam_L_max_new
        else:
            lam_L1 = c_lam_L_max_new

        vals_lamd_L_max.append(lam_L2)
        counterr2 += 1
        if abs(lam_L2 - lam_L1) / lam_L2 < max_lambd_acc:
            break

    c_lambdaN = initi_lambda
    vals_lambda_N = []
    counterr1 = 1
    stopp = 0

    ## finding lambda_N for which matrix of zeros is obtained
    while stopp==0:
        out = spinner_correlation(Y_checked,Z,c_lambdaN, 1e-3)
        if np.linalg.norm(out["B"], "fro") < 1e-16:
            stopp=1
        vals_lambda_N.append(c_lambdaN)
        c_lambdaN = zero_search_ratio * c_lambdaN
        counterr1 += 1

    if len(vals_lambda_N) == 1:
        lam_N1 = 0
        lam_N2 = vals_lambda_N[0]
    else:
        lam_N1 = vals_lambda_N[-2]
        lam_N2 = vals_lambda_N[-1]

    ### finding narrow interval for maximal lambda N
    counterr2 = 1
    vals_lamd_N_max = []
    while True:
        c_lam_L_max_new = (lam_N1 + lam_N2) / 2
        out_new_o =  spinner_correlation(Y_checked,Z,c_lam_L_max_new,1e-3)
        #print("OTO MÓJ COUNTER", counterr2)
        if np.linalg.norm(out_new_o["B"], "fro") < 1e-16:
            lam_N2 = c_lam_L_max_new
        else:
            lam_N1 = c_lam_L_max_new
        vals_lamd_N_max.append(lam_N2)
        counterr2 += 1
        if counterr2>300:
            break
        if abs(lam_N2 - lam_N1) / lam_N2 < max_lambd_acc:
            break

    ### final lambda grids

    k = 0.75
    seqq = np.array([x for x in range(1, grid_lengthL)]) / (grid_lengthL - 1)
    lambs_L_grid = np.insert(np.exp((seqq * np.log(lam_L2 + 1) ** (1 / k)) ** k) - 1, 0, 0.0001)
    lambs_N_grid = np.insert(np.exp((seqq * np.log(lam_N2 + 1) ** (1 / k)) ** k) - 1, 0, 0.0001)

    lambs_L_grid = lambs_L_grid[:10]
    lambs_N_grid = lambs_N_grid[:10]

    print(f"lambs_N_grid {lambs_N_grid}")
    print(f" lambs_L_grid {lambs_L_grid }")



    #logliks_CV = np.zeros((grid_lengthN, grid_lengthL))
    logliks_CV = np.zeros((10, 10))

    min_value = 100000
    row = 0
    for i in tqdm(lambs_N_grid):
        col = 0
        c_lambdaN = max(round(i,2), 1e-4)
        for j in lambs_L_grid:
            #print(f"lambda_N {i} lambda_L {j}")
            c_lambdaL = max(round(j,2), 1e-4)
            norm_res_CV = []
            for g in range(1, kfolds+1):
                test_idx = np.where(groups_idxs==g)
                train_idx =np.where(groups_idxs!=g)

                X_train = X[train_idx,:][0]
                X_test = X[test_idx,:][0]

                X_train = (X_train - np.mean(X_train, axis=0))
                X_test = (X_test - np.mean(X_test, axis=0))

                Y_train = (X_train.T @ X_train) / X_train.shape[0]
                Y_test = (X_test.T @ X_test) / X_test.shape[0]

                out_cv = spinner_correlation(Y_train,Z,c_lambdaN,c_lambdaL)
                #out_cv = spinner_correlation(Y_test, Z, c_lambdaN, c_lambdaL)
                #sns.heatmap(out_cv["B"], center=0, vmin=-1, vmax=1)
                val_for_norm_res = 0.5 * ((out_cv["B"] - Y_test)**2).sum()
                #val_for_norm_res = 0.5 * ((out_cv["B"] - Y_train) ** 2).sum()
                norm_res_CV.append(val_for_norm_res)

            logliks_CV[row, col] = sum(norm_res_CV)/n

            if min_value > sum(norm_res_CV)/n:
                print(f"New minimum value for N {i} and L {j} is {sum(norm_res_CV)/n}")
                min_value = sum(norm_res_CV)/n
            col += 1
        row += 1

    best_idx_lambs = np.where(logliks_CV == np.amin(logliks_CV))
    lambda_N_best_idx = best_idx_lambs[0]
    lambda_L_best_idx = best_idx_lambs[1]

    best_lambda_N = lambs_N_grid[lambda_N_best_idx]
    best_lambda_L = lambs_L_grid[lambda_L_best_idx]


    final_out = spinner_correlation(Y_checked,Z,best_lambda_N,best_lambda_L)

    final_out["bestN"] =  best_lambda_N
    final_out["bestL"] =  best_lambda_L

    return final_out

################# Eksperymenty
# Wejścia do macierzy (-1,1)

# (Czysty strzał, 4, 0.25, 8669)
# (Y_checked,6.64,0.18 8773)
# (Y_test, 0.001, 0.751, 8912)
# (BB, 0.001, 0.751, 8912  )


## Wejścia dla macierzy (-20,20)

# (Czysty strzał, 100, 8,  4451317)
# ( Y_checked, 53, 2.35, 4322308)
# (Y_test, 1.228,14.83, 4490327)
# (BB, 0.001, 14.8 , 4494136  )


if __name__ == "__main__":
    p = 60
    n = 600  # jak było 1000 to było super 500 też
    B1 = 1 * np.ones((15, 15))
    B2 = -1 * np.ones((12, 12))
    B3 = 1 * np.ones((10, 10))
    s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
    left_square = p - s_nods - 18  #
    BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                    np.zeros((left_square, left_square)))
    np.fill_diagonal(BB, 12)
    np.all(np.linalg.eigvals(BB) > 0)

    p = 60
    n = 600
    B1 = 20 * np.ones((15, 15))
    B2 = -20 * np.ones((12, 12))
    B3 = 20 * np.ones((10, 10))
    s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
    left_square = p - s_nods - 18  #
    BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                    np.zeros((left_square, left_square)))
    np.fill_diagonal(BB, 270)
    np.all(np.linalg.eigvals(BB) > 0)

    X = np.zeros((n, p))
    mean = np.zeros(p)
    cov = BB
    np.random.seed(2020)
    for row in range(X.shape[0]):
        X[row, :] = np.random.multivariate_normal(mean, cov, 1)

    Z = np.identity(p)
    Y_checked = (X.T @ X) / X.shape[0]

    final_out = CV_calculate(X)
    final_out_2 = spinner_correlation(Y_checked, Z, 100, 8)

    ((BB - final_out["B"]) ** 2).sum()  # (Y_checked,6.64,0.18 8773) (Y_train, 0.001, 0.751, 8912)
    ((BB - final_out_2["B"]) ** 2).sum()

    sns.heatmap(final_out["B"], center=0, vmin=-20, vmax=20)
    sns.heatmap(final_out_2["B"], center=0, vmin=-5, vmax=5)
    np.unique(final_out["B"])




