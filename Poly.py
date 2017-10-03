import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def r_sq(Y, Y_hat):
    SS_res = (Y - Y_hat).dot(Y - Y_hat)
    SS_tot = (Y - Y.mean()).dot(Y - Y.mean())
    r_squared = 1 - (SS_res / SS_tot)
    return r_squared


data_csv = pd.read_csv("data/data_poly.csv", header=None)
data = data_csv.as_matrix()

Y = data[:, 1]
X = data[:, 0]
bias = np.ones((X.shape[0], 1))
X_sq = X*X
X_sq = np.reshape(X_sq, (X_sq.shape[0], 1))

X = np.reshape(X, (X.shape[0], 1))
X = np.hstack((bias, X))

X = np.hstack((X, X_sq))

w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

print(w)

Y_hat = X.dot(w)

plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Y_hat))
plt.show()

print(Y)
Y=np.reshape(Y, (100,1))
print(Y)

print(r_sq(Y, Y_hat))