import numpy as np
import seaborn as sns
from scipy.linalg import block_diag
import math
import matplotlib.pyplot as plt
import sys
from spinner_corr import spinner_correlation
from tqdm import tqdm
from CV_corr import CV_calculate
import os


def calc_difference(B_estim, B_real):
    numerator = ((B_estim - B_real) ** 2).sum()
    denominator = (B_real ** 2).sum()
    return np.sqrt(numerator) / denominator


p = 60
n = 800  # jak było 1000 to było super 500 też
B1 = 1 * np.ones((15, 15))
B2 = -1 * np.ones((12, 12))
B3 = 1 * np.ones((10, 10))
cubes_dict = {"B1": B1, "B2": B2, "B3": B3}

for cube in cubes_dict.values():
    diag_val = 1
    while np.any(np.linalg.eigvals(cube) < 0):
        np.fill_diagonal(cube, diag_val)
        diag_val += 1
        print(diag_val)
    # np.fill_diagonal(cube, diag_val + 12)

s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
left_square = p - s_nods - 18  #
BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                np.zeros((left_square, left_square))) + np.identity(p) * 5
# sns.heatmap(BB,center=0, vmin=-2, vmax=2)
np.unique(np.diagonal(BB))
X = np.zeros((n, p))
mean = np.zeros(p)
cov = BB
np.random.seed(2020)
for row in range(X.shape[0]):
    X[row, :] = np.random.multivariate_normal(mean, cov, 1)

Z = np.identity(p)
Y_checked = (X.T @ X) / X.shape[0]
sns.heatmap(Y_checked, center=0, vmin=-2, vmax=2)

final_out = CV_calculate(X)
np.fill_diagonal(Y_checked, 0)

print(((BB - final_out["B"]) ** 2).sum())
print(((BB - Y_checked) ** 2).sum())

calc_difference(final_out["B"], BB)
calc_difference(Y_checked, BB)

sns.heatmap(final_out["B"], center=0, vmin=-2, vmax=2)

np.sqrt(((BB - final_out["B"]) ** 2).sum())
np.sqrt(((BB - Y_checked) ** 2).sum())

########                     ########
########  ZWIĘKSZANIE N_KI   ########
########                     ########
p = 60
B1 = 1 * np.ones((15, 15))
B2 = -1 * np.ones((12, 12))
B3 = 1 * np.ones((10, 10))
cubes_dict = {"B1": B1, "B2": B2, "B3": B3}

for cube in cubes_dict.values():
    diag_val = 1
    while np.any(np.linalg.eigvals(cube) < 0):
        np.fill_diagonal(cube, diag_val)
        diag_val += 1
        print(diag_val)
    # np.fill_diagonal(cube, diag_val + 12)

s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
left_square = p - s_nods - 18  #
BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                np.zeros((left_square, left_square))) + np.identity(p) * 5

ols_difference_small_values = []
estim_difference_small_values = []
for n in np.arange(100, 850, 50):
    print(f"CURRENTLY DOING N=={n}")
    X = np.zeros((n, p))
    mean = np.zeros(p)
    cov = BB
    np.random.seed(2020)
    for row in range(X.shape[0]):
        X[row, :] = np.random.multivariate_normal(mean, cov, 1)

    Y_checked = (X.T @ X) / X.shape[0]
    # sns.heatmap(Y_checked, center=0, vmin=-2, vmax=2)
    final_out = CV_calculate(X)
    np.fill_diagonal(Y_checked, 0)

    ols_difference_small_values.append(calc_difference(Y_checked, BB))
    estim_difference_small_values.append(calc_difference(final_out["B"], BB))


plt.plot(np.arange(100, 850, 50), ols_difference_small_values, marker="^", color="blue", label="sample_estimation")
plt.plot(np.arange(100, 850, 50), estim_difference_small_values, marker="v", color="red", label="alg_estimation")
plt.ylabel("Error")
plt.xlabel("Number of rows")
plt.title("Entries +-1")
plt.legend(loc='best')
plt.show()
#plt.savefig("small_val_loss_diff.png")

###########


p = 60
B1 = 20 * np.ones((15, 15))
B2 = -20 * np.ones((12, 12))
B3 = 20 * np.ones((10, 10))
cubes_dict = {"B1": B1, "B2": B2, "B3": B3}

for cube in cubes_dict.values():
    diag_val = 1
    while np.any(np.linalg.eigvals(cube) < 0):
        np.fill_diagonal(cube, diag_val)
        diag_val += 1
        print(diag_val)
    # np.fill_diagonal(cube, diag_val + 12)

s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
left_square = p - s_nods - 18  #
BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                np.zeros((left_square, left_square))) + np.identity(p) * 5

ols_difference_big_values = []
estim_difference_big_values = []
for n in np.arange(100, 850, 50):
    print(f"CURRENTLY DOING N=={n}")
    X = np.zeros((n, p))
    mean = np.zeros(p)
    cov = BB
    np.random.seed(2020)
    for row in range(X.shape[0]):
        X[row, :] = np.random.multivariate_normal(mean, cov, 1)

    Y_checked = (X.T @ X) / X.shape[0]
    # sns.heatmap(Y_checked, center=0, vmin=-2, vmax=2)
    final_out = CV_calculate(X)
    np.fill_diagonal(Y_checked, 0)

    ols_difference_big_values.append(calc_difference(Y_checked, BB))
    estim_difference_big_values.append(calc_difference(final_out["B"], BB))

plt.plot(np.arange(100, 850, 50), ols_difference_big_values, marker="^", color="blue", label="sample_estimation")
plt.plot(np.arange(100, 850, 50), estim_difference_big_values, marker="v", color="red", label="alg_estimation")
plt.ylabel("Error")
plt.xlabel("Number of rows")
plt.legend(loc='best')
plt.title("Entries +- 20")
plt.show()
#plt.savefig("big_val_loss_diff.png")

###########
########### Of diagonal correlation
###########
p = 60

BB = np.zeros((p, p))
B1 = 1 * np.ones((10, 15))
B2 = -1 * np.ones((10, 10))

BB[6:16, 21:36] = B1
BB[21:36, 6:16] = B1.T
BB[40:50, 40:50] = B2

# sns.heatmap(BB,center=0, vmin=-2, vmax=2)
diag = 0.1
while np.any(np.linalg.eigvals(BB) < 0):
    np.fill_diagonal(BB, diag)
    diag += 0.1
# sns.heatmap(BB,center=0, vmin=-2, vmax=2)

off_daigonal_ols_difference_small_values = []
off_diagonal_estim_difference_small_values = []
for n in np.arange(100, 850, 50):
    print(f"CURRENTLY DOING N=={n}")
    X = np.zeros((n, p))
    mean = np.zeros(p)
    cov = BB
    np.random.seed(2020)
    for row in range(X.shape[0]):
        X[row, :] = np.random.multivariate_normal(mean, cov, 1)

    Y_checked = (X.T @ X) / X.shape[0]

    final_out = CV_calculate(X)
    np.fill_diagonal(Y_checked, 0)

    off_daigonal_ols_difference_small_values.append(calc_difference(Y_checked, BB))
    off_diagonal_estim_difference_small_values.append(calc_difference(final_out["B"], BB))

plt.plot(np.arange(100, 850, 50), off_daigonal_ols_difference_small_values, marker="^", color="blue",
         label="sample_estimation")
plt.plot(np.arange(100, 850, 50), off_diagonal_estim_difference_small_values, marker="v", color="red",
         label="alg_estimation")
plt.ylabel("Error")
plt.xlabel("Number of rows")
plt.legend(loc='best')
plt.title("Entries +- 1")
plt.show()
#plt.savefig("small_val_off_diagonal_loss_diff.png")

###### BIG
p = 60

BB = np.zeros((p, p))
B1 = 20 * np.ones((10, 15))
B2 = -20 * np.ones((10, 10))

BB[6:16, 21:36] = B1
BB[21:36, 6:16] = B1.T
BB[40:50, 40:50] = B2

diag = 1
while np.any(np.linalg.eigvals(BB) < 0):
    np.fill_diagonal(BB, diag)
    diag += 1

# sns.heatmap(BB,center=0, vmin=-20, vmax=20)

off_daigonal_ols_difference_big_values = []
off_diagonal_estim_difference_big_values = []
for n in np.arange(100, 850, 50):
    print(f"CURRENTLY DOING N=={n}")

    X = np.zeros((n, p))
    mean = np.zeros(p)
    cov = BB
    np.random.seed(2020)
    for row in range(X.shape[0]):
        X[row, :] = np.random.multivariate_normal(mean, cov, 1)

    Y_checked = (X.T @ X) / X.shape[0]
    final_out = CV_calculate(X)
    np.fill_diagonal(Y_checked, 0)

    off_daigonal_ols_difference_big_values.append(calc_difference(Y_checked, BB))
    off_diagonal_estim_difference_big_values.append(calc_difference(final_out["B"], BB))

plt.plot(np.arange(100, 850, 50), off_daigonal_ols_difference_big_values, marker="^", color="blue",
         label="sample_estimation")
plt.plot(np.arange(100, 850, 50), off_diagonal_estim_difference_big_values, marker="v", color="red",
         label="alg_estimation")
plt.ylabel("Error")
plt.xlabel("Number of rows")
plt.legend(loc='best')
plt.title("Entries +- 20")
plt.show()
#plt.savefig("big_val_off_diagonal_loss_diff.png")

###############
###############
###############
############### Poszczególne bloczki ###############
###############
###############
###############

lost_B1_ols_100n_bigsquare_smalval = []
lost_B2_ols_100n_bigsquare_smalval = []
lost_B3_ols_100n_bigsquare_smalval = []

lost_B1_estim_100n_bigsquare_smalval = []
lost_B2_estim_100n_bigsquare_smalval = []
lost_B3_estim_100n_bigsquare_smalval = []

loss_ols_100n_biqsuare_smalval = []
loss_estim_100n_bigsquare_smalval = []

list_of_beta_estim_100n_bigsquare_smalval = []
list_of_beta_ols_100n_bigsquare_smalval = []

for b1_val in np.arange(0.1, 20, 1):
    print(f"---------Current b1_val is {b1_val}------------")
    n = 100
    p = 60
    if b1_val > 1:
        b1_val = round(b1_val)
    B1 = b1_val * np.ones((15, 15))
    B2 = -1 * np.ones((12, 12))
    B3 = 1 * np.ones((10, 10))
    cubes_dict = {"B1": B1, "B2": B2, "B3": B3}

    for cube in cubes_dict.values():
        diag_val = 1
        while np.any(np.linalg.eigvals(cube) < 0):
            np.fill_diagonal(cube, diag_val)
            diag_val += 1
            print(diag_val)

    s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
    left_square = p - s_nods - 18  #
    BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                    np.zeros((left_square, left_square))) + np.identity(p) * 5

    B1_prim = BB[5:20, 5:20]
    B2_prim = BB[26:38, 26:38]
    B3_prim = BB[45:55, 45:55]

    X = np.zeros((n, p))
    mean = np.zeros(p)
    cov = BB
    np.random.seed(2020)
    for row in range(X.shape[0]):
        X[row, :] = np.random.multivariate_normal(mean, cov, 1)

    Y_checked = (X.T @ X) / X.shape[0]
    np.fill_diagonal(Y_checked, 0)
    list_of_beta_ols_100n_bigsquare_smalval.append(Y_checked)

    B1_ols = Y_checked[5:20, 5:20]
    B2_ols = Y_checked[26:38, 26:38]
    B3_ols = Y_checked[45:55, 45:55]

    lost_B1_ols_100n_bigsquare_smalval.append(calc_difference(B1_ols, B1_prim))
    lost_B2_ols_100n_bigsquare_smalval.append(calc_difference(B2_ols, B2_prim))
    lost_B3_ols_100n_bigsquare_smalval.append(calc_difference(B3_ols, B3_prim))

    # sns.heatmap(Y_checked,center=0, vmin=-2, vmax=2)
    final_out = CV_calculate(X)
    B_estim = final_out["B"]
    list_of_beta_estim_100n_bigsquare_smalval.append(B_estim)

    B1_estim = B_estim[5:20, 5:20]
    B2_estim = B_estim[26:38, 26:38]
    B3_estim = B_estim[45:55, 45:55]

    lost_B1_estim_100n_bigsquare_smalval.append(calc_difference(B1_estim, B1_prim))
    lost_B2_estim_100n_bigsquare_smalval.append(calc_difference(B2_estim, B2_prim))
    lost_B3_estim_100n_bigsquare_smalval.append(calc_difference(B3_estim, B3_prim))

    loss_ols_100n_biqsuare_smalval.append(calc_difference(Y_checked, BB))
    loss_estim_100n_bigsquare_smalval.append(calc_difference(final_out["B"], BB))

plt.plot(np.arange(0.1, 20, 1), lost_B1_estim_100n_bigsquare_smalval, color="blue", label="B1_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B2_estim_100n_bigsquare_smalval, color="red", label="B2_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B3_estim_100n_bigsquare_smalval, color="green", label="B3_estim")
plt.ylabel("Error")
plt.xlabel("val in  B1")
plt.legend(loc='best')
plt.title("Recovery of block B1")
plt.show()

plt.plot(np.arange(0.1, 20, 1), lost_B1_ols_100n_bigsquare_smalval, color="blue", label="B1_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B2_ols_100n_bigsquare_smalval, color="red", label="B2_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B3_ols_100n_bigsquare_smalval, color="green", label="B3_estim")
plt.ylabel("Error")
plt.xlabel("val in  B1")
plt.legend(loc='best')
plt.title("Recovery of block B1")
plt.show()

plt.plot(np.arange(0.1, 20, 1), lost_B1_estim_100n_bigsquare_smalval, color="blue", label="B1_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B1_ols_100n_bigsquare_smalval, color="red", label="B1_ols")
plt.legend(loc="best")

plt.plot(np.arange(0.1, 20, 1), lost_B2_estim_100n_bigsquare_smalval, color="blue", label="B2_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B2_ols_100n_bigsquare_smalval, color="red", label="B2_ols")
plt.legend(loc="best")

plt.plot(np.arange(0.1, 20, 1), lost_B3_estim_100n_bigsquare_smalval, color="blue", label="B3_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B3_ols_100n_bigsquare_smalval, color="red", label="B3_ols")
plt.legend(loc="best")


fig = plt.figure(figsize=(5,8))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

sns.heatmap(list_of_beta_ols_100n_bigsquare_smalval[0], center=0, vmin=-1, vmax=1, ax = ax1)
sns.heatmap(list_of_beta_ols_100n_bigsquare_smalval[4], center=0, vmin=-1, vmax=1, ax = ax2)
sns.heatmap(list_of_beta_ols_100n_bigsquare_smalval[10], center=0, vmin=-1, vmax=1, ax = ax3)

plt.savefig("ols_100_n_bigsquare_smalval.png")


fig = plt.figure(figsize=(5,8))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

sns.heatmap(list_of_beta_estim_100n_bigsquare_smalval[0], center=0, vmin=-1, vmax=1, ax= ax1)
sns.heatmap(list_of_beta_estim_100n_bigsquare_smalval[4], center=0, vmin=-1, vmax=1, ax = ax2)
sns.heatmap(list_of_beta_estim_100n_bigsquare_smalval[10], center=0, vmin=-1, vmax=1, ax = ax3)

plt.savefig("estim_100_n_bigsquare_smalval.png")


plt.plot(np.arange(0.1, 20, 1), loss_ols_100n_biqsuare_smalval, color = "red", label="ols")
plt.plot(np.arange(0.1, 20, 1),loss_estim_100n_bigsquare_smalval, color = "blue",label = "estim")
plt.legend(loc="best")
plt.show()

plt.savefig("loss_100_n_bigsquare_smalval.png")

# B33 = list_of_beta_estim_100n_bigsquare_smalval[-1][45:55, 45:55]





###############################

lost_B1_ols_300n_bigsquare_smalval = []
lost_B2_ols_300n_bigsquare_smalval = []
lost_B3_ols_300n_bigsquare_smalval = []

lost_B1_estim_300n_bigsquare_smalval = []
lost_B2_estim_300n_bigsquare_smalval = []
lost_B3_estim_300n_bigsquare_smalval = []

loss_ols_300n_biqsuare_smalval = []
loss_estim_300n_bigsquare_smalval = []

list_of_beta_estim_300n_bigsquare_smalval = []
list_of_beta_ols_300n_bigsquare_smalval = []

for b1_val in np.arange(0.1, 20, 1):
    n = 300
    p = 60
    if b1_val > 1:
        b1_val = round(b1_val)
        print(f"---------Current b1_val is {b1_val}------------")
    B1 = b1_val * np.ones((15, 15))
    B2 = -1 * np.ones((12, 12))
    B3 = 1 * np.ones((10, 10))
    cubes_dict = {"B1": B1, "B2": B2, "B3": B3}

    for cube in cubes_dict.values():
        diag_val = 1
        while np.any(np.linalg.eigvals(cube) < 0):
            np.fill_diagonal(cube, diag_val)
            diag_val += 1
            print(diag_val)

    s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
    left_square = p - s_nods - 18  #
    BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                    np.zeros((left_square, left_square))) + np.identity(p) * 5

    B1_prim = BB[5:20, 5:20]
    B2_prim = BB[26:38, 26:38]
    B3_prim = BB[45:55, 45:55]

    X = np.zeros((n, p))
    mean = np.zeros(p)
    cov = BB
    np.random.seed(2020)
    for row in range(X.shape[0]):
        X[row, :] = np.random.multivariate_normal(mean, cov, 1)

    Y_checked = (X.T @ X) / X.shape[0]
    np.fill_diagonal(Y_checked, 0)
    list_of_beta_ols_300n_bigsquare_smalval.append(Y_checked)

    B1_ols = Y_checked[5:20, 5:20]
    B2_ols = Y_checked[26:38, 26:38]
    B3_ols = Y_checked[45:55, 45:55]

    lost_B1_ols_300n_bigsquare_smalval.append(calc_difference(B1_ols, B1_prim))
    lost_B2_ols_300n_bigsquare_smalval.append(calc_difference(B2_ols, B2_prim))
    lost_B3_ols_300n_bigsquare_smalval.append(calc_difference(B3_ols, B3_prim))

    # sns.heatmap(Y_checked,center=0, vmin=-2, vmax=2)
    final_out = CV_calculate(X)
    B_estim = final_out["B"]
    list_of_beta_estim_300n_bigsquare_smalval.append(B_estim)

    B1_estim = B_estim[5:20, 5:20]
    B2_estim = B_estim[26:38, 26:38]
    B3_estim = B_estim[45:55, 45:55]

    lost_B1_estim_300n_bigsquare_smalval.append(calc_difference(B1_estim, B1_prim))
    lost_B2_estim_300n_bigsquare_smalval.append(calc_difference(B2_estim, B2_prim))
    lost_B3_estim_300n_bigsquare_smalval.append(calc_difference(B3_estim, B3_prim))

    loss_ols_300n_biqsuare_smalval.append(calc_difference(Y_checked, BB))
    loss_estim_300n_bigsquare_smalval.append(calc_difference(final_out["B"], BB))


############################
############################  500
############################
###########################

lost_B1_ols_500n_bigsquare_smalval = []
lost_B2_ols_500n_bigsquare_smalval = []
lost_B3_ols_500n_bigsquare_smalval = []

lost_B1_estim_500n_bigsquare_smalval = []
lost_B2_estim_500n_bigsquare_smalval = []
lost_B3_estim_500n_bigsquare_smalval = []

loss_ols_500n_biqsuare_smalval = []
loss_estim_500n_bigsquare_smalval = []

list_of_beta_estim_500n_bigsquare_smalval = []
list_of_beta_ols_500n_bigsquare_smalval = []

for b1_val in np.arange(0.1, 20, 1):
    print(f"---------Current b1_val is {b1_val}------------")
    n = 500
    p = 60
    B1 = b1_val * np.ones((15, 15))
    B2 = -1 * np.ones((12, 12))
    B3 = 1 * np.ones((10, 10))
    cubes_dict = {"B1": B1, "B2": B2, "B3": B3}

    for cube in cubes_dict.values():
        diag_val = 1
        while np.any(np.linalg.eigvals(cube) < 0):
            np.fill_diagonal(cube, diag_val)
            diag_val += 1
            print(diag_val)

    s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
    left_square = p - s_nods - 18  #
    BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                    np.zeros((left_square, left_square))) + np.identity(p) * 5

    B1_prim = BB[5:20, 5:20]
    B2_prim = BB[26:38, 26:38]
    B3_prim = BB[45:55, 45:55]

    X = np.zeros((n, p))
    mean = np.zeros(p)
    cov = BB
    np.random.seed(2020)
    for row in range(X.shape[0]):
        X[row, :] = np.random.multivariate_normal(mean, cov, 1)

    Y_checked = (X.T @ X) / X.shape[0]
    np.fill_diagonal(Y_checked, 0)
    list_of_beta_ols_500n_bigsquare_smalval.append(Y_checked)

    B1_ols = Y_checked[5:20, 5:20]
    B2_ols = Y_checked[26:38, 26:38]
    B3_ols = Y_checked[45:55, 45:55]

    lost_B1_ols_500n_bigsquare_smalval.append(calc_difference(B1_ols, B1_prim))
    lost_B2_ols_500n_bigsquare_smalval.append(calc_difference(B2_ols, B2_prim))
    lost_B3_ols_500n_bigsquare_smalval.append(calc_difference(B3_ols, B3_prim))

    # sns.heatmap(Y_checked,center=0, vmin=-2, vmax=2)
    final_out = CV_calculate(X)
    B_estim = final_out["B"]
    list_of_beta_estim_500n_bigsquare_smalval.append(B_estim)

    B1_estim = B_estim[5:20, 5:20]
    B2_estim = B_estim[26:38, 26:38]
    B3_estim = B_estim[45:55, 45:55]

    lost_B1_estim_500n_bigsquare_smalval.append(calc_difference(B1_estim, B1_prim))
    lost_B2_estim_500n_bigsquare_smalval.append(calc_difference(B2_estim, B2_prim))
    lost_B3_estim_500n_bigsquare_smalval.append(calc_difference(B3_estim, B3_prim))

    loss_ols_500n_biqsuare_smalval.append(calc_difference(Y_checked, BB))
    loss_estim_500n_bigsquare_smalval.append(calc_difference(final_out["B"], BB))


plt.plot( np.arange(0.1, 20, 1), lost_B1_estim_500n_bigsquare_smalval, color="blue", label="B1_estim")
plt.plot( np.arange(0.1, 20, 1), lost_B2_estim_500n_bigsquare_smalval, color="red", label="B2_estim")
plt.plot( np.arange(0.1, 20, 1), lost_B3_estim_500n_bigsquare_smalval, color="green", label="B3_estim")
plt.ylabel("Error")
plt.xlabel("val in  B1")
plt.legend(loc='best')
plt.title("Recovery of block B1")
plt.show()


plt.plot(np.arange(0.1, 20, 1), lost_B1_estim_500n_bigsquare_smalval, color="blue", label="B1_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B1_ols_500n_bigsquare_smalval, color="red", label="B1_ols")

plt.plot(np.arange(0.1, 20, 1), lost_B2_estim_500n_bigsquare_smalval, color="blue", label="B2_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B2_ols_500n_bigsquare_smalval, color="red", label="B2_ols")

plt.plot(np.arange(0.1, 20, 1), lost_B3_estim_500n_bigsquare_smalval, color="blue", label="B3_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B3_ols_500n_bigsquare_smalval, color="red", label="B3_ols")

plt.plot(np.arange(0.1, 20, 1), loss_ols_500n_biqsuare_smalval, color = "blue", label = "ols")
plt.plot(np.arange(0.1, 20, 1),loss_estim_500n_bigsquare_smalval, color = "red", label = "estim")
plt.legend(loc="best")
plt.savefig("500_n_bigsquare_smalval_lost.png")

len(list_of_beta_ols_500n_bigsquare_smalval)[20]

B1_ols = Y_checked[5:20, 5:20]
B2_ols = Y_checked[26:38, 26:38]
B3_ols = Y_checked[45:55, 45:55]


fig = plt.figure(figsize=(5,8))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

sns.heatmap(list_of_beta_ols_500n_bigsquare_smalval[0],center=0, vmin=-1, vmax=1, ax = ax1)
sns.heatmap(list_of_beta_ols_500n_bigsquare_smalval[4],center=0, vmin=-1, vmax=1, ax = ax2)
sns.heatmap(list_of_beta_ols_500n_bigsquare_smalval[10],center=0, vmin=-1, vmax=1, ax = ax3)
plt.savefig("ols_500_n_bigsquare_smalval.png")



fig = plt.figure(figsize=(5,8))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

sns.heatmap(list_of_beta_estim_500n_bigsquare_smalval[0],center=0, vmin=-1, vmax=1, ax = ax1)
sns.heatmap(list_of_beta_estim_500n_bigsquare_smalval[4],center=0, vmin=-1, vmax=1, ax = ax2)
sns.heatmap(list_of_beta_estim_500n_bigsquare_smalval[10],center=0, vmin=-1, vmax=1, ax = ax3)
plt.savefig("estim_500_n_bigsquare_smalval.png")



########
######## UJEMNY BLOCZEK
########

lost_B1_ols_100n_negativesquare_smalval = []
lost_B2_ols_100n_negativesquare_smalval = []
lost_B3_ols_100n_negativesquare_smalval = []

lost_B1_estim_100n_negativesquare_smalval = []
lost_B2_estim_100n_negativesquare_smalval = []
lost_B3_estim_100n_negativesquare_smalval = []

loss_ols_100n_negativesquare_smalval = []
loss_estim_100n_negativesquare_smalval = []

list_of_beta_estim_100n_negativesquare_smalval = []
list_of_beta_ols_100n_negativesquare_smalval = []

for b2_val in np.arange(0.1, 20, 1):
    print(f"---------Current b1_val is {b2_val}------------")
    n = 500
    p = 60
    if b1_val > 1:
        b1_val = round(b1_val)
    B1 = 1 * np.ones((15, 15))
    B2 = -b2_val * np.ones((12, 12))
    B3 = 1 * np.ones((10, 10))
    cubes_dict = {"B1": B1, "B2": B2, "B3": B3}

    for cube in cubes_dict.values():
        diag_val = 1
        while np.any(np.linalg.eigvals(cube) < 0):
            np.fill_diagonal(cube, diag_val)
            diag_val += 1
            print(diag_val)

    s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
    left_square = p - s_nods - 18  #
    BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                    np.zeros((left_square, left_square))) + np.identity(p) * 5

    B1_prim = BB[5:20, 5:20]
    B2_prim = BB[26:38, 26:38]
    B3_prim = BB[45:55, 45:55]

    X = np.zeros((n, p))
    mean = np.zeros(p)
    cov = BB
    np.random.seed(2020)
    for row in range(X.shape[0]):
        X[row, :] = np.random.multivariate_normal(mean, cov, 1)

    Y_checked = (X.T @ X) / X.shape[0]
    np.fill_diagonal(Y_checked, 0)
    list_of_beta_ols_100n_negativesquare_smalval.append(Y_checked)

    B1_ols = Y_checked[5:20, 5:20]
    B2_ols = Y_checked[26:38, 26:38]
    B3_ols = Y_checked[45:55, 45:55]

    lost_B1_ols_100n_negativesquare_smalval.append(calc_difference(B1_ols, B1_prim))
    lost_B2_ols_100n_negativesquare_smalval.append(calc_difference(B2_ols, B2_prim))
    lost_B3_ols_100n_negativesquare_smalval.append(calc_difference(B3_ols, B3_prim))

    # sns.heatmap(Y_checked,center=0, vmin=-2, vmax=2)
    final_out = CV_calculate(X)
    B_estim = final_out["B"]
    list_of_beta_estim_100n_negativesquare_smalval.append(B_estim)

    B1_estim = B_estim[5:20, 5:20]
    B2_estim = B_estim[26:38, 26:38]
    B3_estim = B_estim[45:55, 45:55]

    lost_B1_estim_100n_negativesquare_smalval.append(calc_difference(B1_estim, B1_prim))
    lost_B2_estim_100n_negativesquare_smalval.append(calc_difference(B2_estim, B2_prim))
    lost_B3_estim_100n_negativesquare_smalval.append(calc_difference(B3_estim, B3_prim))

    loss_ols_100n_negativesquare_smalval.append(calc_difference(Y_checked, BB))
    loss_estim_100n_negativesquare_smalval.append(calc_difference(final_out["B"], BB))



plt.plot(np.arange(0.1, 20, 1), lost_B1_estim_100n_negativesquare_smalval, color="blue", label="B1_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B2_estim_100n_negativesquare_smalval, color="red", label="B2_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B3_estim_100n_negativesquare_smalval, color="green", label="B3_estim")
plt.ylabel("Error")
plt.xlabel("val in  B1")
plt.legend(loc='best')
plt.title("Recovery of block B1")
plt.show()



plt.plot(np.arange(0.1, 20, 1), lost_B1_estim_100n_negativesquare_smalval, color="blue", label="B1_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B1_ols_100n_negativesquare_smalval, color="red", label="B1_ols")
plt.legend(loc="best")

plt.plot(np.arange(0.1, 20, 1), lost_B2_estim_100n_negativesquare_smalval, color="blue", label="B2_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B2_ols_100n_negativesquare_smalval, color="red", label="B2_ols")
plt.legend(loc="best")

plt.plot(np.arange(0.1, 20, 1), lost_B3_estim_100n_negativesquare_smalval, color="blue", label="B3_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B3_ols_100n_negativesquare_smalval, color="red", label="B3_ols")
plt.legend(loc="best")


fig = plt.figure(figsize=(5,8))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

sns.heatmap(list_of_beta_ols_100n_negativesquare_smalval[0], center=0, vmin=-1, vmax=1, ax = ax1)
sns.heatmap(list_of_beta_ols_100n_negativesquare_smalval[4], center=0, vmin=-1, vmax=1, ax = ax2)
sns.heatmap(list_of_beta_ols_100n_negativesquare_smalval[10], center=0, vmin=-1, vmax=1, ax = ax3)

plt.savefig("ols_500_n_negativesquare_smalval.png")


fig = plt.figure(figsize=(5,8))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

sns.heatmap(list_of_beta_estim_100n_negativesquare_smalval[0], center=0, vmin=-1, vmax=1, ax= ax1)
sns.heatmap(list_of_beta_estim_100n_negativesquare_smalval[4], center=0, vmin=-1, vmax=1, ax = ax2)
sns.heatmap(list_of_beta_estim_100n_negativesquare_smalval[10], center=0, vmin=-1, vmax=1, ax = ax3)

plt.savefig("estim_500_n_negativesquare_smalval.png")


plt.plot(np.arange(0.1, 20, 1), loss_ols_100n_negativesquare_smalval, color = "red", label="ols")
plt.plot(np.arange(0.1, 20, 1),loss_estim_100n_negativesquare_smalval, color = "blue",label = "estim")
plt.legend(loc="best")
plt.show()

plt.savefig("loss_500_n_negativesquare_smalval.png")



###################
###################
################### NAJMNIEJSZY BLOCZEK
###################
###################
###################

lost_B1_ols_500n_smallestsquare_smalval = []
lost_B2_ols_500n_smallestsquare_smalval = []
lost_B3_ols_500n_smallestsquare_smalval = []

lost_B1_estim_500n_smallestsquare_smalval = []
lost_B2_estim_500n_smallestsquare_smalval = []
lost_B3_estim_500n_smallestsquare_smalval = []

loss_ols_500n_smallestsquare_smalval = []
loss_estim_500n_smallestsquare_smalval = []

list_of_beta_estim_500n_smallestsquare_smalval = []
list_of_beta_ols_500n_smallestsquare_smalval = []


for b3_val in np.arange(0.1, 20, 1):
    print(f"---------Current b3_val is {b3_val}------------")
    n = 400
    p = 60
    if b3_val > 1:
        b3_val = round(b3_val)
    B1 = 1 * np.ones((15, 15))
    B2 = -1 * np.ones((12, 12))
    B3 = b3_val * np.ones((10, 10))
    cubes_dict = {"B1": B1, "B2": B2, "B3": B3}

    for cube in cubes_dict.values():
        diag_val = 1
        while np.any(np.linalg.eigvals(cube) < 0):
            np.fill_diagonal(cube, diag_val)
            diag_val += 1
            print(diag_val)

    s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
    left_square = p - s_nods - 18  #
    BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                    np.zeros((left_square, left_square))) + np.identity(p) * 5

    B1_prim = BB[5:20, 5:20]
    B2_prim = BB[26:38, 26:38]
    B3_prim = BB[45:55, 45:55]

    X = np.zeros((n, p))
    mean = np.zeros(p)
    cov = BB
    np.random.seed(2020)
    for row in range(X.shape[0]):
        X[row, :] = np.random.multivariate_normal(mean, cov, 1)

    Y_checked = (X.T @ X) / X.shape[0]
    np.fill_diagonal(Y_checked, 0)
    list_of_beta_ols_500n_smallestsquare_smalval.append(Y_checked)

    B1_ols = Y_checked[5:20, 5:20]
    B2_ols = Y_checked[26:38, 26:38]
    B3_ols = Y_checked[45:55, 45:55]

    lost_B1_ols_500n_smallestsquare_smalval.append(calc_difference(B1_ols, B1_prim))
    lost_B2_ols_500n_smallestsquare_smalval.append(calc_difference(B2_ols, B2_prim))
    lost_B3_ols_500n_smallestsquare_smalval.append(calc_difference(B3_ols, B3_prim))

    # sns.heatmap(Y_checked,center=0, vmin=-2, vmax=2)
    final_out = CV_calculate(X)
    B_estim = final_out["B"]
    list_of_beta_estim_500n_smallestsquare_smalval.append(B_estim)

    B1_estim = B_estim[5:20, 5:20]
    B2_estim = B_estim[26:38, 26:38]
    B3_estim = B_estim[45:55, 45:55]

    lost_B1_estim_500n_smallestsquare_smalval.append(calc_difference(B1_estim, B1_prim))
    lost_B2_estim_500n_smallestsquare_smalval.append(calc_difference(B2_estim, B2_prim))
    lost_B3_estim_500n_smallestsquare_smalval.append(calc_difference(B3_estim, B3_prim))

    loss_ols_500n_smallestsquare_smalval.append(calc_difference(Y_checked, BB))
    loss_estim_500n_smallestsquare_smalval.append(calc_difference(final_out["B"], BB))



plt.plot(np.arange(0.1, 20, 1), lost_B1_estim_500n_smallestsquare_smalval, color="blue", label="B1_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B2_estim_500n_smallestsquare_smalval, color="red", label="B2_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B3_estim_500n_smallestsquare_smalval, color="green", label="B3_estim")
plt.ylabel("Error")
plt.xlabel("val in  B1")
plt.legend(loc='best')
plt.title("Recovery of block B1")
plt.show()



plt.plot(np.arange(0.1, 20, 1), lost_B1_estim_500n_smallestsquare_smalval, color="blue", label="B1_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B1_ols_500n_smallestsquare_smalval, color="red", label="B1_ols")
plt.legend(loc="best")

plt.plot(np.arange(0.1, 20, 1), lost_B2_estim_500n_smallestsquare_smalval, color="blue", label="B2_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B2_ols_500n_smallestsquare_smalval, color="red", label="B2_ols")
plt.legend(loc="best")

plt.plot(np.arange(0.1, 20, 1), lost_B3_estim_500n_smallestsquare_smalval, color="blue", label="B3_estim")
plt.plot(np.arange(0.1, 20, 1), lost_B3_ols_500n_smallestsquare_smalval, color="red", label="B3_ols")
plt.legend(loc="best")


fig = plt.figure(figsize=(5,8))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

sns.heatmap(list_of_beta_ols_500n_smallestsquare_smalval[0], center=0, vmin=-1, vmax=1, ax = ax1)
sns.heatmap(list_of_beta_ols_500n_smallestsquare_smalval[4], center=0, vmin=-1, vmax=1, ax = ax2)
sns.heatmap(list_of_beta_ols_500n_smallestsquare_smalval[10], center=0, vmin=-1, vmax=1, ax = ax3)

plt.savefig("ols_500_n_smallestsquare_smalval.png")


fig = plt.figure(figsize=(5,8))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

sns.heatmap(list_of_beta_estim_500n_smallestsquare_smalval[0], center=0, vmin=-1, vmax=1, ax= ax1)
sns.heatmap(list_of_beta_estim_500n_smallestsquare_smalval[4], center=0, vmin=-1, vmax=1, ax = ax2)
sns.heatmap(list_of_beta_estim_500n_smallestsquare_smalval[10], center=0, vmin=-1, vmax=1, ax = ax3)

plt.savefig("estim_500_n_smallestsquare_smalval.png")


plt.plot(np.arange(0.1, 20, 1), loss_ols_500n_smallestsquare_smalval, color = "red", label="ols")
plt.plot(np.arange(0.1, 20, 1),loss_estim_500n_smallestsquare_smalval, color = "blue",label = "estim")
plt.legend(loc="best")
plt.show()

plt.savefig("loss_500_n_smallestsquare_smalval.png")


import pickle

def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True

bk = {}
for k in dir():
    obj = globals()[k]
    if is_picklable(obj):
        try:
            bk.update({k: obj})
        except TypeError:
            pass

# to save session
with open('./your_bk.pkl', 'wb') as f:
    pickle.dump(bk, f)

# with open('./your_bk.pkl', 'rb') as f:
#     bk_restore = pickle.load(f)