import numpy as np
import seaborn as sns
from scipy.linalg import block_diag
from CV_corr import CV_calculate

p = 60 # kolumny
n = 800 # wiersze

# tworzę bloczki do macierzy korelacji
B1 = 1 * np.ones((15, 15))
B2 = -1 * np.ones((12, 12))
B3 = 1 * np.ones((10, 10))
cubes_dict = {"B1": B1, "B2": B2, "B3": B3}

# upewniam się że każdy z tych 3 bloczków jest positive semidefine
for cube in cubes_dict.values():
    diag_val = 1
    while np.any(np.linalg.eigvals(cube) < 0): # to jest warunek na semidefine
        np.fill_diagonal(cube, diag_val)
        diag_val += 1
        print(diag_val)

# Tworzenie właściwej macierzy korelacji gdzie wkładam moje bloczki
s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
left_square = p - s_nods - 18  #
BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                np.zeros((left_square, left_square))) + np.identity(p) * 5



sns.heatmap(BB,center=0, vmin=-2, vmax=2) # można obejrzeć wyjściową macierz korealcji


# Wytworzenie danych X o macierzy korelacji BB
X = np.zeros((n, p))
mean = np.zeros(p)
cov = BB
np.random.seed(2020)
for row in range(X.shape[0]):
    X[row, :] = np.random.multivariate_normal(mean, cov, 1)

Z = np.identity(p) # to nie ma znacznia (tricki do kodu)
Y_checked = (X.T @ X) / X.shape[0] # próbkowa macierz kowariancji
sns.heatmap(Y_checked, center=0, vmin=-2, vmax=2)

final_out = CV_calculate(X) # MÓJ SPINNER, nie przejmuj się ostrzeżeniami na początku
np.fill_diagonal(Y_checked, 0) # żeby porównać estymacje tutaj trzeba uzupełnić zerami
sns.heatmap(final_out["B"], center=0, vmin=-2, vmax=2)

# przykładowe wyniki
print(((BB - final_out["B"]) ** 2).sum())
print(((BB - Y_checked) ** 2).sum())

