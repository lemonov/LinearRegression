import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def r_sq(y, y_hat):
    ss_res = (y - y_hat).dot(y - y_hat)
    ss_tot = (y - y.mean()).dot(y - y.mean())
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def get_r(x, y):
    w = np.linalg.solve(x.T.dot(x), x.T.dot(y))
    y_hat = x.dot(w)
    return r_sq(y, y_hat)


data_xls = pd.read_excel("data/mlr02.xls")
data_xls['ones'] = 1

random = np.random.randn(data_xls['X1'].shape[0], 1)

Y = data_xls['X1']
X = data_xls[['X2', 'X3', 'ones']]
X = np.hstack((X, random))
X2_only = data_xls[['X2', 'ones']]
X3_only = data_xls[['X3', 'ones']]
X2_only = np.hstack((X2_only, random))
X3_only = np.hstack((X3_only, random))



print(type(X))
print(type(X2_only))
print(type(X3_only))

print(get_r(X, Y))
print(get_r(X2_only, Y))
print(get_r(X3_only, Y))

w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

y_hat = X.dot(w)
# plot data
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
ax.plot_trisurf(X[:, 0], X[:, 1], y_hat)
plt.show()

X = np.array([28, 211, 1, 0])
print("My blood pressure", X.dot(w))
