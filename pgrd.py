import  numpy as np
import seaborn as sns
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import sys

p = 60
n = 100
B1 = 2 * np.ones((15,15))
B2 = -2 * np.ones((12,12))
B3 = 2 * np.ones((10,10))
s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
left_square = p - s_nods - 18 #
B = block_diag(np.zeros((5,5)), B1, np.zeros((6,6)), B2, np.zeros((7,7)),B3, np.zeros((left_square,left_square)) )
np.fill_diagonal(B,23)
np.all(np.linalg.eigvals(B)>0)
sns.heatmap(B, center=0, vmin=-2.5, vmax=2.5)
# B_corr = (B - np.mean(B,axis=0)) / np.std(B, axis=0)
# sns.heatmap((B - np.mean(B,axis=0)) / np.std(B, axis=0), center=0, vmin=-10, vmax=10)

# p = 20
# n = 50
# B1 = 2 * np.ones((5,5))
# B2 = -2 * np.ones((3,3))
# B3 = 2 * np.ones((3,3))
# s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
# left_square = p - s_nods - 6 #
# BB = block_diag(np.zeros((2,2)), B1, np.zeros((2,2)), B2, np.zeros((2,2)),B3, np.zeros((left_square,left_square)) )
# np.fill_diagonal(BB,10)
# np.all(np.linalg.eigvals(BB)>0)
# sns.heatmap(B, center=0, vmin=-2.5, vmax=2.5)










X = np.zeros((n,p))
mean = np.zeros(p)
cov = B
for row in range(X.shape[0]):
    X[row,:] = np.random.multivariate_normal(mean, cov, 1)

sns.heatmap(X, center=0, vmin=-20, vmax=20)

X = (X - np.mean(X,axis=0)) / np.std(X, axis=0)
Y= X.T @ X
sns.heatmap(Y, center=0, vmin=-20, vmax=20)
#########
##########
########## TUTAJ ZAPRASZAM
##########
##########
##########
from spinner_corr import spinner_correlation
#np.set_printoptions(threshold=sys.maxsize)
p = 60
n = 100
B1 = 1 * np.ones((15, 15))
B2 = -1 * np.ones((12, 12))
B3 = 1 * np.ones((10, 10))
s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
left_square = p - s_nods - 18  #
BB = block_diag(np.zeros((5, 5)), B1, np.zeros((6, 6)), B2, np.zeros((7, 7)), B3,
                np.zeros((left_square, left_square)))
np.fill_diagonal(BB, 12)
np.all(np.linalg.eigvals(BB) > 0)
sns.heatmap(BB, center=0, vmin=-1, vmax=1)


p = 60
n = 100
B1 = 5 * np.ones((15, 15))
B2 = -1 * np.ones((15, 15))
B3 = 5 * np.ones((15, 15))
s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
left_square = max(p - s_nods - 18,0)  #
BB = block_diag(np.zeros((4, 4)), B1, np.zeros((6, 6)), B2, np.zeros((5, 5)), B3,
                np.zeros((left_square, left_square)))
np.fill_diagonal(BB, 15)
np.all(np.linalg.eigvals(BB) > 0)
sns.heatmap(BB, center=0, vmin=-1, vmax=15)

p = 60
n = 100
B1 = 1 * np.ones((15, 15))
B2 = -5 * np.ones((15, 15))
B3 = 1 * np.ones((15, 15))
s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
left_square = max(p - s_nods - 18,0)  #
BB = block_diag(np.zeros((4, 4)), B1, np.zeros((6, 6)), B2, np.zeros((5, 5)), B3,
                np.zeros((left_square, left_square)))
np.fill_diagonal(BB, 75)
np.all(np.linalg.eigvals(BB) > 0)
sns.heatmap(BB, center=0, vmin=-10, vmax=10)




X = np.zeros((n, p))
mean = np.zeros(p)
cov = BB
np.random.seed(2020)
for row in range(X.shape[0]):
    X[row, :] = np.random.multivariate_normal(mean, cov, 1)


#X = np.random.multivariate_normal(mean, cov, n)
###############
###############
###############
# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
#Y = X.T @ X
#Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
# Y = Y - np.diag(Y)

X = (X - np.mean(X, axis=0))
Y_checked = (X.T @X)/X.shape[0]
sns.heatmap(Y_checked , center=0, vmin=-5, vmax=5)

put_to_diagonal = np.diagonal(Y_checked).copy()
np.fill_diagonal(Y_checked,0)
sns.heatmap(Y_checked , center=0, vmin=-2, vmax=2)

out = spinner_correlation(Y_checked, Z, 7, 2)
sns.heatmap(out["B"], center=0, vmin=-0.5, vmax=0.5)
B_estim = out["B"]
np.fill_diagonal(B_estim,put_to_diagonal)
((B_estim - BB)**2).sum()


sns.heatmap(np.cov(X.T), center=0, vmin=-5, vmax=5)
np.fill_diagonal(Y, 0)
sns.heatmap(Y, center=0, vmin=-1, vmax=1)
Z = np.identity(Y.shape[0])
lambdaN = 2
lambdaL = 2

Y_magic = BB + np.random.randn(p, p)
np.fill_diagonal(Y_magic, 0)
out = spinner_correlation(Y_magic, Z, 3, 0.3)
sns.heatmap(out["B"], center=0, vmin=-2, vmax=2)

Y

Y_cor_numpy = np.cov(X.T)
np.fill_diagonal(Y_cor_numpy,0)
sns.heatmap(Y_cor_numpy , center=0, vmin=-5, vmax=5)
out = spinner_correlation(Y_cor_numpy, Z, 6, 0.8)
sns.heatmap(out["B"], center=0, vmin=-5, vmax=5)

out = spinner_correlation(Y, Z, 5, 8)

sns.heatmap(out["B"], center=0, vmin=-20, vmax=20)

sns.heatmap(BB, center=0, vmin=-2, vmax=2)

############
############
### WIĘksze wymiary
p = 60
n = 600
B1 = 5 * np.ones((15,15))
B2 = -5 * np.ones((12,12))
B3 = 5 * np.ones((10,10))
s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
left_square = p - s_nods - 18 #
BB = block_diag(np.zeros((5,5)), B1, np.zeros((6,6)), B2, np.zeros((7,7)),B3, np.zeros((left_square,left_square)) )
np.fill_diagonal(BB,80)
np.all(np.linalg.eigvals(BB)>0)
sns.heatmap(BB, center=0, vmin=-5, vmax=5)

Z = np.identity(p)

X = np.zeros((n,p))
mean = np.zeros(p)
cov = BB
for row in range(X.shape[0]):
    X[row,:] = np.random.multivariate_normal(mean, cov, 1)

X = (X - np.mean(X, axis=0))
Y_checked = (X.T @X)/X.shape[0]
put_to_diagonal = np.diagonal(Y_checked).copy()
np.fill_diagonal(Y_checked,0)
sns.heatmap(Y_checked , center=0, vmin=-10, vmax=10)

out = spinner_correlation(Y_checked , Z, 15, 2.5)
sns.heatmap(out["B"], center=0, vmin=-2, vmax=2)

Y_magic_big1 = BB + np.random.randn(p, p)
np.fill_diagonal(Y_magic_big1, 0)
out = spinner_correlation(Y_magic_big1, Z, 3, 0.3)
sns.heatmap(out["B"], center=0, vmin=-2, vmax=2)

sns.heatmap(Y_magic_big1, center=0, vmin=-10, vmax=10)