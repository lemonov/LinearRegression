import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# import data
data_csv = pd.read_csv("data/data_2d.csv", header=None)
data = data_csv.as_matrix()

Y = data[:, 2]
bias = np.ones((Y.shape[0],1))

X = np.hstack((bias, data[:, 0:2]))

W = np.linalg.solve((X.T.dot(X)), X.T.dot(Y))

print(W)

Y_hat = X.dot(W)

# plot data
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X[:, 1], X[:, 2], Y)
ax.plot_trisurf(X[:, 1], X[:, 2], Y_hat)
plt.show()


def r_sq(Y, Y_hat):
    SS_res = (Y - Y_hat).dot(Y - Y_hat)
    SS_tot = (Y - Y.mean()).dot(Y - Y.mean())
    r_squared = 1 - (SS_res / SS_tot)
    return r_squared


print(r_sq(Y, Y_hat))
