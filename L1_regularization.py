import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 50
D = 50

X = (np.random.random((N, D)) - 0.5) * 10

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3)) # append vector to D-3 0-es

Y = X.dot(true_w) + np.random.randn(N) * 0.5

costs = []

w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 10.0
for i in range(0, 500, 1):
    y_hat = X.dot(w)
    delta = y_hat - Y
    w = w - learning_rate *(X.T.dot(delta) + l1 * np.sign(w))
    costs.append(delta.dot(delta)/N)

plt.plot(costs)
plt.show()

plt.plot(true_w, label= "true_w")
plt.plot(w, label= "w")
plt.legend()
plt.show()
# plot data
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(X[0], X[1], Y)
# plt.show()